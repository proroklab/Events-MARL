from __future__ import annotations

import numpy as np


def compute_gae(rewards, values, dones, *, gamma: float, gae_lambda: float, reward_scale: float,
                num_envs: int, num_agents: int, current_num_agents: int,
                advantage_clip_range: float, mixed_agent_training: bool):
    rewards = np.asarray(rewards) * reward_scale
    num_steps = len(rewards)
    values = np.asarray(values)

    values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
    values = np.clip(values, -100.0, 100.0)
    rewards = rewards.reshape(num_steps, num_envs, num_agents)
    rewards = np.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
    rewards = np.clip(rewards, -10.0, 10.0)

    dones_per_env = np.any(
        np.asarray(dones).reshape(num_steps, num_envs, num_agents), axis=-1
    ).astype(float)

    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    gae = np.zeros((num_envs, num_agents))
    next_value = np.zeros((num_envs, num_agents))

    for t in reversed(range(num_steps)):
        next_non_terminal = (1.0 - dones_per_env[t])[:, np.newaxis]
        next_val = next_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_non_terminal * next_val - values[t]
        delta = np.clip(delta, -100.0, 100.0)
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        gae = np.clip(gae, -100.0, 100.0)
        advantages[t] = gae
        returns[t] = gae + values[t]

    if mixed_agent_training:
        agent_mask = np.arange(num_agents) < current_num_agents
        advantages = advantages * agent_mask[np.newaxis, np.newaxis, :]

    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)

    if mixed_agent_training:
        active = advantages_flat.reshape(-1, num_agents)[:, :current_num_agents].flatten()
        mean, std = active.mean(), active.std()
    else:
        mean, std = advantages_flat.mean(), advantages_flat.std()

    advantages_flat = (advantages_flat - mean) / (std + 1e-8)
    advantages_flat = np.clip(advantages_flat, -advantage_clip_range, advantage_clip_range)
    return advantages_flat, returns_flat, values.reshape(-1)


def normalize_returns(returns_batch, old_values):
    import jax.numpy as jnp

    mean, std = returns_batch.mean(), returns_batch.std()
    return (jnp.asarray((returns_batch - mean) / (std + 1e-8)),
            (old_values - float(mean)) / (float(std) + 1e-8))

