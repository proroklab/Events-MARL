from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Requery:
    indices: np.ndarray
    inputs: Any
    entries: list = field(default_factory=list)
    scaling: float | None = None
    tokens: Any = None
    alpha: float | None = None


class NoEvents:
    def __init__(self, num_envs: int, num_agents: int):
        self._zeros = torch.zeros(num_envs, num_agents, dtype=torch.bool)

    def prime(self, reset_obs) -> None:
        return None

    def before_step(self, env, obs, tokens, step: int) -> Requery | None:
        return None

    def after_step(self, env, next_obs, tokens, step: int) -> Requery | None:
        return None

    def commit(self) -> None:
        return None

