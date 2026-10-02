from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MinibatchIndices:
    obs_indices: jnp.ndarray
    env_indices: jnp.ndarray
    minibatch_size: int
    minibatch_env_size: int

    def slice(self, index: int):
        start, end = index * self.minibatch_size, (index + 1) * self.minibatch_size
        env_start = index * self.minibatch_env_size
        env_end = env_start + self.minibatch_env_size
        return self.obs_indices[start:end], self.env_indices[env_start:env_end]


def make_indices(total_agent_rows: int, total_env_rows: int, num_agents: int,
                 num_minibatches: int, shuffle: bool, shuffle_rng) -> MinibatchIndices:
    if shuffle and shuffle_rng is not None:
        env_indices = jax.random.permutation(shuffle_rng, jnp.arange(total_env_rows))
        agent_offsets = jnp.arange(num_agents)
        obs_indices = (env_indices[:, None] * num_agents + agent_offsets[None, :]).flatten()
    else:
        env_indices = jnp.arange(total_env_rows)
        obs_indices = jnp.arange(total_agent_rows)
    return MinibatchIndices(
        obs_indices=obs_indices,
        env_indices=env_indices,
        minibatch_size=total_agent_rows // num_minibatches,
        minibatch_env_size=total_env_rows // num_minibatches,
    )


def ps_minibatch_indices(total_rows: int, num_minibatches: int, shuffle: bool, shuffle_rng):
    if total_rows % num_minibatches:
        raise ValueError(f"batch of {total_rows} rows is not divisible into {num_minibatches}")
    indices = jnp.arange(total_rows)
    if shuffle and shuffle_rng is not None:
        indices = jax.random.permutation(shuffle_rng, indices)
    size = total_rows // num_minibatches
    return [indices[i * size:(i + 1) * size] for i in range(num_minibatches)]

