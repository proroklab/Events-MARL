from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2, 3, 4))
def calculate_snd_dico(policy_params, obs_batch, policy_model, num_agents: int, num_envs: int,
                       diversity_scaling=1.0):
    actual_num_envs = obs_batch.shape[0] // num_agents
    agent_ids = jnp.tile(jnp.arange(num_agents), actual_num_envs)

    mean_batch, _ = policy_model.apply(
        {"params": policy_params}, obs_batch, agent_ids, diversity_scaling)
    means = mean_batch.reshape(actual_num_envs, num_agents, mean_batch.shape[-1])

    if num_agents < 2:
        return jnp.array(0.0)

    i_pairs, j_pairs = jnp.triu_indices(num_agents, k=1)

    def w2(i, j):
        return jnp.sqrt(jnp.mean(jnp.sum((means[:, i, :] - means[:, j, :]) ** 2, axis=-1)))

    return jnp.mean(jax.vmap(w2)(i_pairs, j_pairs))


def dico_scaling(target_snd: float, ema, max_scaling: float, min_snd_floor: float):
    scaling = target_snd / jnp.maximum(ema, min_snd_floor)
    scaling = jnp.clip(scaling, 0.0001, max_scaling)
    return jnp.where(jnp.isnan(scaling) | jnp.isinf(scaling), 1.0, scaling)

