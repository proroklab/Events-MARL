from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch

from ..tokens import HypernetworkInputs, extract_lidar, lidar_dim
from .base import Requery

FOOD_IN_RANGE_INDEX = 7


def detect_food_in_range(obs_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([agent_obs[:, FOOD_IN_RANGE_INDEX] > 0.5 for agent_obs in obs_list], dim=1)


class NavigationEvents:
    def __init__(self, num_envs: int, num_agents: int, adaptive: bool, obs_dim: int = 0,
                 device: str = "cpu"):
        self.adaptive = adaptive
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.already_detected = torch.zeros(num_envs, num_agents, dtype=torch.bool, device=device)

    def prime(self, reset_obs: list[torch.Tensor]) -> None:
        if self.adaptive:
            self.already_detected = detect_food_in_range(reset_obs).clone()

    def after_step(self, env, next_obs, tokens, step: int) -> Requery | None:
        newly = self.newly_detected(next_obs)
        if not bool(newly.any()):
            return None
        indices = np.where(newly.reshape(-1).cpu().numpy())[0]
        inputs, entries = self.requery_inputs(next_obs, indices, tokens)
        return Requery(indices=indices, inputs=inputs, entries=entries)

    def newly_detected(self, next_obs: list[torch.Tensor]) -> torch.Tensor:
        if not self.adaptive:
            return torch.zeros_like(self.already_detected)
        current = detect_food_in_range(next_obs)
        newly = current & ~self.already_detected
        self._current = current
        return newly

    def before_step(self, env, obs, tokens, step: int) -> Requery | None:
        return None

    def commit(self) -> None:
        if self.adaptive:
            self.already_detected = self.already_detected | self._current

    def requery_inputs(self, next_obs, indices, tokens) -> tuple[HypernetworkInputs, list[dict]]:
        batch_size = next_obs[0].shape[0] * self.num_agents
        updated = extract_lidar(next_obs, lidar_dim(self.obs_dim)).reshape(batch_size, -1)
        updated = np.nan_to_num(updated.cpu().numpy(), posinf=1.0, neginf=0.0)
        updated = jnp.asarray(np.clip(updated, -1.0, 1.0))
        subset = updated[indices][:, None, :]

        entries = [{"task": None, "context": None, "lidar": subset[i, 0], "target_snd": None,
                    "query_type": "requery", "hidden_state": None,
                    "agent_idx": int(idx) % self.num_agents}
                   for i, idx in enumerate(indices)]
        return HypernetworkInputs(lidar=subset, mask=None), entries

