from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import torch


@dataclass
class Trajectory:
    obs: list = field(default_factory=list)
    global_states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    token_rows: list = field(default_factory=list)
    episode_rewards: list = field(default_factory=list)
    requery_events: list = field(default_factory=list)
    num_resets: int = 0


@dataclass
class RolloutResult:
    trajectory: Trajectory
    obs: list
    adapters: dict
    tokens: object
    alpha: float


def _apply_requery(requery, adapters, tokens, hn_apply, alpha, buffer, traj, step):
    if requery is None:
        return adapters, tokens, alpha
    tokens = requery.tokens if requery.tokens is not None else tokens
    if requery.alpha is not None:
        alpha = requery.alpha
    if len(requery.indices) == 0:
        return adapters, tokens, alpha
    scaling = alpha if requery.scaling is None else requery.scaling
    new_adapters = hn_apply(requery.inputs, scaling)
    adapters = {k: adapters[k].at[requery.indices].set(new_adapters[k]) for k in adapters}
    if requery.tokens is None:
        tokens = tokens.with_rows(requery.indices, requery.inputs)
    buffer.extend(requery.entries)
    traj.requery_events.append((step, requery.indices.tolist()))
    return adapters, tokens, alpha


def _global_state(obs_transposed: torch.Tensor, num_envs: int, num_agents: int,
                  current_num_agents: int, mixed: bool) -> torch.Tensor:
    if mixed and current_num_agents < num_agents:
        active = obs_transposed[:, :current_num_agents, :]
        padding = torch.zeros(num_envs, num_agents - current_num_agents, obs_transposed.shape[-1],
                              dtype=obs_transposed.dtype, device=obs_transposed.device)
        active = torch.cat([active, padding], dim=1)
        return active.reshape(num_envs, -1)
    return obs_transposed.reshape(num_envs, -1)


def collect_rollout(*, env, obs, policy_params, critic_params, adapters, hn_apply, tokens, events,
                    config, num_agents, current_num_agents, episode: int, alpha: float,
                    buffer: list, get_actions, get_value, obs_suffix=None) -> RolloutResult:
    env_config, ppo = config.env, config.ppo
    num_envs = env_config.num_envs
    batch_size = num_envs * num_agents
    mixed = env_config.mixed_agent_training
    traj = Trajectory()

    jax_rng = jax.random.PRNGKey(episode)

    for step in range(ppo.rollout_steps):
        adapters, tokens, alpha = _apply_requery(events.before_step(env, obs, tokens, step),
                                                 adapters, tokens, hn_apply, alpha, buffer, traj,
                                                 step)

        obs_transposed = torch.stack(obs, dim=0).transpose(0, 1)
        jax_obs = jnp.asarray(obs_transposed.reshape(batch_size, -1).cpu().numpy())
        if obs_suffix is not None:
            jax_obs = jnp.concatenate([jax_obs, obs_suffix], axis=-1)

        global_state = _global_state(obs_transposed, num_envs, num_agents, current_num_agents,
                                     mixed)
        jax_global_state = jnp.asarray(global_state.cpu().numpy())
        if ppo.clip_global_state:
            jax_global_state = jnp.clip(jax_global_state, ppo.clip_global_state_min,
                                        ppo.clip_global_state_max)
        values = get_value(critic_params, jax_global_state)

        jax_rng, action_rng = jax.random.split(jax_rng)
        action, log_prob, _hidden, _mean, _std = get_actions(
            policy_params, jax_obs, adapters, action_rng)

        actions_flat = np.asarray(action)
        if np.isnan(actions_flat).any():
            actions_flat = np.nan_to_num(actions_flat, nan=0.0, posinf=0.0, neginf=0.0)
        actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
        if mixed:
            agent_mask = np.arange(num_agents) < current_num_agents
            actions_reshaped = actions_reshaped * agent_mask[np.newaxis, :, np.newaxis]

        torch_actions = [
            torch.tensor(actions_reshaped[:, i, :], dtype=torch.float32) for i in range(num_agents)
        ]
        next_obs, rewards, dones, _info = env.step(torch_actions)

        traj.obs.append(jax_obs)
        traj.global_states.append(jax_global_state)
        traj.actions.append(action)
        traj.token_rows.append(tokens.flatten_rows(batch_size))

        if mixed:
            for agent_idx in range(current_num_agents, num_agents):
                rewards[agent_idx] = torch.zeros_like(rewards[agent_idx])

        adapters, tokens, alpha = _apply_requery(events.after_step(env, next_obs, tokens, step),
                                                 adapters, tokens, hn_apply, alpha, buffer, traj,
                                                 step)
        events.commit()

        traj.log_probs.append(jnp.clip(
            jnp.nan_to_num(log_prob, nan=-100.0, posinf=-100.0, neginf=-100.0), -100.0, 100.0))
        traj.values.append(jnp.clip(
            jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0))

        rewards_np = torch.stack(rewards, dim=0).transpose(0, 1).cpu().numpy()
        rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=0.0, neginf=0.0)
        if mixed:
            rewards_np = rewards_np * (np.arange(num_agents) < current_num_agents)[np.newaxis, :]
        traj.rewards.append(np.asarray(rewards_np).flatten())
        reward_mean = float(rewards_np.mean())
        traj.episode_rewards.append(0.0 if not np.isfinite(reward_mean) else reward_mean)

        traj.dones.append(jnp.array(np.repeat(dones.cpu().numpy(), num_agents), dtype=bool))

        if bool(dones.all().item()):
            traj.num_resets += 1
            next_obs = env.reset()

        obs = next_obs

    return RolloutResult(trajectory=traj, obs=obs, adapters=adapters, tokens=tokens,
                         alpha=alpha)


@dataclass
class RecurrentTrajectory(Trajectory):
    init_hidden: object = None
    init_critic_hidden: object = None


def collect_rollout_recurrent(*, env, obs, policy_params, critic_params, capabilities, config,
                              num_agents, current_num_agents, episode: int, get_actions,
                              get_value, hidden_dim: int, critic_hidden_dim: int,
                              obs_suffix=None) -> RolloutResult:
    env_config, ppo = config.env, config.ppo
    num_envs = env_config.num_envs
    batch_size = num_envs * num_agents
    mixed = env_config.mixed_agent_training
    traj = RecurrentTrajectory()

    hidden = jnp.zeros((batch_size, hidden_dim))
    critic_hidden = jnp.zeros((num_envs, critic_hidden_dim))
    traj.init_hidden, traj.init_critic_hidden = hidden, critic_hidden

    jax_rng = jax.random.PRNGKey(episode)
    dones = None

    for step in range(ppo.rollout_steps):
        prev_dones = (jnp.zeros(batch_size, dtype=bool) if step == 0
                      else jnp.asarray(traj.dones[-1]))

        obs_transposed = torch.stack(obs, dim=0).transpose(0, 1)
        jax_obs = jnp.asarray(obs_transposed.reshape(batch_size, -1).cpu().numpy())
        if obs_suffix is not None:
            jax_obs = jnp.concatenate([jax_obs, obs_suffix], axis=-1)

        global_state = _global_state(obs_transposed, num_envs, num_agents, current_num_agents,
                                     mixed)
        jax_global_state = jnp.asarray(global_state.cpu().numpy())
        if ppo.clip_global_state:
            jax_global_state = jnp.clip(jax_global_state, ppo.clip_global_state_min,
                                        ppo.clip_global_state_max)

        env_dones = torch.zeros(num_envs, dtype=torch.bool)
        if step > 0:
            for agent_idx in range(len(dones)):
                env_dones = env_dones | dones[agent_idx]
        critic_hidden, values = get_value(critic_params, jax_global_state, critic_hidden,
                                          jnp.asarray(env_dones.cpu().numpy()))

        jax_rng, action_rng = jax.random.split(jax_rng)
        action, log_prob, new_hidden, _mean, _std = get_actions(
            policy_params, jax_obs, capabilities, action_rng, hidden, prev_dones)
        hidden = new_hidden

        actions_flat = np.asarray(action)
        if np.isnan(actions_flat).any():
            actions_flat = np.nan_to_num(actions_flat, nan=0.0, posinf=0.0, neginf=0.0)
        actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
        if mixed:
            agent_mask = np.arange(num_agents) < current_num_agents
            actions_reshaped = actions_reshaped * agent_mask[np.newaxis, :, np.newaxis]

        torch_actions = [
            torch.tensor(actions_reshaped[:, i, :], dtype=torch.float32) for i in range(num_agents)
        ]
        next_obs, rewards, dones, _info = env.step(torch_actions)

        traj.obs.append(jax_obs)
        traj.global_states.append(jax_global_state)
        traj.actions.append(action)

        if mixed:
            for agent_idx in range(current_num_agents, num_agents):
                rewards[agent_idx] = torch.zeros_like(rewards[agent_idx])

        traj.log_probs.append(jnp.clip(
            jnp.nan_to_num(log_prob, nan=-100.0, posinf=-100.0, neginf=-100.0), -100.0, 100.0))
        traj.values.append(jnp.clip(
            jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0))

        rewards_np = torch.stack(rewards, dim=0).transpose(0, 1).cpu().numpy()
        rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=0.0, neginf=0.0)
        if mixed:
            rewards_np = rewards_np * (np.arange(num_agents) < current_num_agents)[np.newaxis, :]
        traj.rewards.append(np.asarray(rewards_np).flatten())
        reward_mean = float(rewards_np.mean())
        traj.episode_rewards.append(0.0 if not np.isfinite(reward_mean) else reward_mean)

        traj.dones.append(jnp.array(np.repeat(dones.cpu().numpy(), num_agents), dtype=bool))

        if bool(dones.all().item()):
            traj.num_resets += 1
            next_obs = env.reset()

        obs = next_obs

    return RolloutResult(trajectory=traj, obs=obs, adapters=None, tokens=None, alpha=1.0)

