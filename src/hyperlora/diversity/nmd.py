from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def wasserstein_distance_gaussian(mean1, std1, mean2, std2, just_mean: bool = True):
    mean_dist = jnp.linalg.norm(mean1 - mean2, ord=2, axis=-1)
    if just_mean:
        return mean_dist
    cov_dist = jnp.linalg.norm(std1 - std2, ord=2, axis=-1)
    return jnp.sqrt(mean_dist**2 + cov_dist**2)


def compute_nmd_from_action_means(action_means, action_stds=None, just_mean: bool = True) -> float:
    num_adapters = action_means.shape[0]
    if num_adapters < 2:
        return 0.0
    if action_stds is None:
        action_stds = jnp.ones_like(action_means) * 1e-8
    i_pairs, j_pairs = jnp.triu_indices(num_adapters, k=1)

    def pair_distance(i, j):
        dist = wasserstein_distance_gaussian(action_means[i], action_stds[i],
                                             action_means[j], action_stds[j], just_mean=just_mean)
        return jnp.mean(dist)

    return float(jnp.mean(jax.vmap(pair_distance)(i_pairs, j_pairs)))


def _sample_entries(buffer, sample_size: int, rng_key):
    if len(buffer) <= sample_size:
        return list(buffer), rng_key
    rng_key, subkey = jax.random.split(rng_key)
    indices = jax.random.choice(subkey, len(buffer), shape=(sample_size,), replace=False)
    return [buffer[int(i)] for i in np.array(indices)], rng_key


def _sample_observations(trajectory_obs, sample_size: int, rng_key):
    all_obs = jnp.concatenate(list(trajectory_obs), axis=0)
    total_obs = all_obs.shape[0]
    num_eval_obs = min(sample_size, total_obs)
    if num_eval_obs < total_obs:
        rng_key, subkey = jax.random.split(rng_key)
        obs_indices = jax.random.choice(subkey, total_obs, shape=(num_eval_obs,), replace=False)
        return all_obs[obs_indices], num_eval_obs, rng_key
    return all_obs, num_eval_obs, rng_key


def wind_flocking_adapter_positions(num_adapters: int, num_agents: int) -> jnp.ndarray:
    seed = int(time.time() * 1000000) % (2**32)
    key = jax.random.PRNGKey(seed)

    spacing = 0.3
    base_x = jnp.arange(num_agents) * spacing - (num_agents - 1) * spacing / 2
    formations = []
    for _ in range(num_adapters):
        key, subkey = jax.random.split(key)
        x_offset = jax.random.uniform(subkey, shape=(), minval=-1.0, maxval=1.0)
        key, subkey = jax.random.split(key)
        y_offset = jax.random.uniform(subkey, shape=(), minval=-0.5, maxval=0.5)
        key, subkey = jax.random.split(key)
        x_noise = jax.random.normal(subkey, shape=(num_agents,)) * 0.1
        key, subkey = jax.random.split(key)
        y_noise = jax.random.normal(subkey, shape=(num_agents,)) * 0.1

        positions = jnp.stack(
            [base_x + x_offset + x_noise, jnp.ones(num_agents) * y_offset + y_noise], axis=1)
        formations.append(positions - jnp.mean(positions, axis=0))
    return jnp.stack(formations, axis=0)


def calculate_nmd(
    buffer: Sequence[Mapping[str, Any]],
    sample_size: int,
    hn_params: Any,
    policy_params: Any,
    hypernetwork,
    policy_model,
    rng_key,
    trajectory_obs: Sequence[jnp.ndarray] | None,
    diversity_scaling: float = 1.0,
    agent_positions_fn=None,
) -> float:
    if not buffer:
        return 0.0
    if not trajectory_obs:
        return 0.0

    entries, rng_key = _sample_entries(buffer, sample_size, rng_key)
    num_adapters = len(entries)
    if num_adapters == 0:
        return 0.0
    obs_batch, num_eval_obs, rng_key = _sample_observations(trajectory_obs, sample_size, rng_key)

    def _stack(key, width_if_missing=0):
        values = [e.get(key) for e in entries]
        if values[0] is None:
            return jnp.zeros((num_adapters, width_if_missing)) if width_if_missing == 0 else None
        return jnp.stack(values, axis=0)

    task_batch = _stack("task")
    context_batch = _stack("context")
    lidar_batch = None if entries[0].get("lidar") is None else jnp.stack(
        [e["lidar"] for e in entries], axis=0)

    hn_max_agents = hypernetwork.max_agents
    repeat = lambda x: None if x is None else jnp.repeat(x[:, None, :], hn_max_agents, axis=1)
    zeros = lambda dim: (jnp.zeros((num_adapters, hn_max_agents, dim)) if dim > 0 else None)

    agent_positions = zeros(hypernetwork.agent_position_dim)
    if hypernetwork.agent_position_dim > 0 and agent_positions_fn is not None:
        generated = agent_positions_fn(num_adapters)
        if generated.shape[1] < hn_max_agents:
            pad = jnp.zeros((num_adapters, hn_max_agents - generated.shape[1],
                             hypernetwork.agent_position_dim))
            generated = jnp.concatenate([generated, pad], axis=1)
        agent_positions = generated[:, :hn_max_agents, :]

    adapters_all = hypernetwork.apply(
        {"params": hn_params},
        repeat(task_batch),
        repeat(context_batch),
        repeat(lidar_batch),
        zeros(hypernetwork.food_position_dim),
        agent_positions,
        zeros(hypernetwork.target_nmd_dim),
        zeros(hypernetwork.env_context_dim),
        mask=None,
        diversity_scaling=diversity_scaling,
    )

    agent_indices = jnp.array([e.get("agent_idx", 0) for e in entries])
    adapters = {}
    for key, value in adapters_all.items():
        reshaped = value.reshape(num_adapters, hn_max_agents, *value.shape[1:])
        adapters[key] = jax.vmap(lambda a, i: a[i])(reshaped, agent_indices)

    obs_tiled = jnp.tile(obs_batch, (num_adapters, 1))
    adapters_tiled = {k: jnp.repeat(v, num_eval_obs, axis=0) for k, v in adapters.items()}
    means, log_stds = policy_model.apply({"params": policy_params}, obs_tiled, adapters_tiled)

    action_dim = means.shape[-1]
    means = means.reshape(num_adapters, num_eval_obs, action_dim)
    stds = jnp.exp(log_stds).reshape(num_adapters, num_eval_obs, action_dim)
    return compute_nmd_from_action_means(means, stds)


def calculate_nmd_dispersion(
    buffer: Sequence[Mapping[str, Any]],
    sample_size: int,
    hn_params: Any,
    policy_params: Any,
    hypernetwork,
    policy_model,
    num_agents: int,
    rng_key,
    trajectory_obs: Sequence[jnp.ndarray] | None,
    diversity_scaling: float = 1.0,
) -> float:
    if not buffer or not trajectory_obs:
        return 0.0

    _entries, rng_key = _sample_entries(buffer, sample_size, rng_key)
    _obs, _count, rng_key = _sample_observations(trajectory_obs, sample_size, rng_key)

    hn_max_agents = hypernetwork.max_agents
    obs_dim = trajectory_obs[0].shape[-1]
    num_envs_traj = trajectory_obs[0].shape[0] // num_agents
    num_steps = len(trajectory_obs)

    grouped = jnp.stack(list(trajectory_obs), axis=0).reshape(
        num_steps, num_envs_traj, num_agents, obs_dim)

    total_pairs = num_steps * num_envs_traj
    actual_sample_size = min(sample_size, total_pairs)
    rng_key, subkey = jax.random.split(rng_key)
    pair_indices = jax.random.choice(subkey, total_pairs, shape=(actual_sample_size,),
                                     replace=actual_sample_size > total_pairs)
    sampled = grouped[pair_indices // num_envs_traj, pair_indices % num_envs_traj]

    food = jnp.concatenate(
        [sampled[:, :, 4 + k * 3: 4 + k * 3 + 2] for k in range(num_agents)], axis=-1)
    if num_agents < hn_max_agents:
        food = jnp.concatenate(
            [food, jnp.zeros((actual_sample_size, hn_max_agents - num_agents, food.shape[-1]))],
            axis=1)

    agent_positions = None
    if hypernetwork.agent_position_dim > 0:
        agent_positions = sampled[:, :, :2]
        if num_agents < hn_max_agents:
            agent_positions = jnp.concatenate(
                [agent_positions,
                 jnp.zeros((actual_sample_size, hn_max_agents - num_agents, 2))], axis=1)

    context_dim = hypernetwork.context_dim
    if context_dim > 0:
        context = jnp.broadcast_to(jnp.eye(hn_max_agents, context_dim)[None, :, :],
                                   (actual_sample_size, hn_max_agents, context_dim))
    else:
        context = jnp.zeros((actual_sample_size, hn_max_agents, context_dim))

    zeros = lambda dim: (jnp.zeros((actual_sample_size, hn_max_agents, dim)) if dim > 0 else None)
    adapters_all = hypernetwork.apply(
        {"params": hn_params},
        jnp.zeros((actual_sample_size, hn_max_agents, hypernetwork.task_embed_dim)),
        context,
        None,
        food,
        agent_positions,
        zeros(hypernetwork.target_nmd_dim),
        zeros(hypernetwork.env_context_dim),
        mask=None,
        diversity_scaling=diversity_scaling,
    )

    adapters = {k: v.reshape(actual_sample_size, hn_max_agents, *v.shape[1:])[:, :num_agents]
                for k, v in adapters_all.items()}
    common_obs = jnp.broadcast_to(sampled[:, 0:1, :], (actual_sample_size, num_agents, obs_dim))
    means, log_stds = policy_model.apply(
        {"params": policy_params},
        common_obs.reshape(-1, obs_dim),
        {k: v.reshape(-1, *v.shape[2:]) for k, v in adapters.items()},
    )
    action_dim = means.shape[-1]
    means = jnp.transpose(means.reshape(actual_sample_size, num_agents, action_dim), (1, 0, 2))
    stds = jnp.transpose(
        jnp.exp(log_stds).reshape(actual_sample_size, num_agents, action_dim), (1, 0, 2))
    return compute_nmd_from_action_means(means, stds)

