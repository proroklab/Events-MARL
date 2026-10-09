from __future__ import annotations

import numpy as np

from ...diversity.control import (alpha_broadcast, alpha_from_ema, alpha_mean_over_envs,
                                  scheduled_target)
from ..tokens import wind_flocking_inputs
from .base import Requery


class WindFlockingEvents:
    def __init__(self, config, num_envs: int, num_agents: int, episode: int, ema: float):
        self.config = config
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.episode = episode
        self.ema = ema
        diversity = config.diversity
        self.targets = tuple(diversity.target_nmd_list)
        self.interval = diversity.target_nmd_change_interval
        self.enabled = bool(self.targets) and self.interval > 0
        if self.enabled and diversity.target_nmd_mode != "cycle":
            raise NotImplementedError(
                "only the 'cycle' target-NMD schedule is supported")

    def prime(self, reset_obs) -> None:
        return None

    def commit(self) -> None:
        return None

    def after_step(self, env, next_obs, tokens, step: int) -> Requery | None:
        return None

    def before_step(self, env, obs, tokens, step: int) -> Requery | None:
        if not self.enabled or step <= 0 or step % self.interval != 0:
            return None

        target = scheduled_target(self.episode, step, self.config.ppo.rollout_steps,
                                  self.targets, self.interval)
        updated = wind_flocking_inputs(self.config, obs, self.num_envs, self.num_agents, target)
        alpha = alpha_from_ema(target, self.ema, self.config.diversity)
        indices = np.arange(self.num_envs * self.num_agents)
        return Requery(
            indices=indices,
            inputs=updated,
            entries=[],
            scaling=alpha_broadcast(alpha, self.num_envs, self.num_agents),
            tokens=updated,
            alpha=alpha_mean_over_envs(alpha, self.num_envs),
        )

