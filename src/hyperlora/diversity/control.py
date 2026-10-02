from __future__ import annotations

import jax.numpy as jnp

from ..config import Config, DiversityConfig


def update_ema(current_ema: float, measured: float, coef: float) -> float:
    return (1 - coef) * current_ema + coef * measured


def alpha_from_ema(target_snd: float, ema: float, config: DiversityConfig) -> float:
    alpha = float(jnp.sqrt(target_snd / jnp.maximum(ema, config.min_snd_floor)))
    return float(jnp.clip(alpha, 0.001, jnp.sqrt(config.max_diversity_scaling)))


def alpha_mean_over_envs(alpha: float, num_envs: int) -> float:
    return float(jnp.mean(jnp.full(num_envs, alpha)))


def alpha_broadcast(alpha: float, num_envs: int, num_agents: int) -> jnp.ndarray:
    per_env = jnp.full(num_envs, alpha)
    return jnp.repeat(per_env, num_agents)[:, None, None]


def scheduled_target(episode: int, step: int, rollout_steps: int, targets: tuple,
                     interval: int) -> float:
    return float(targets[((episode * rollout_steps + step) // interval) % len(targets)])


def episode_target(config: Config, episode: int) -> float:
    diversity = config.diversity
    if not (diversity.target_snd_list and diversity.target_snd_change_interval > 0):
        return diversity.target_snd
    if diversity.target_snd_mode != "cycle":
        raise NotImplementedError(
            "only the 'cycle' target-NMD schedule is supported")
    return scheduled_target(episode, 0, config.ppo.rollout_steps,
                            tuple(diversity.target_snd_list),
                            diversity.target_snd_change_interval)

