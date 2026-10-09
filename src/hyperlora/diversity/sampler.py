from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetNMDSampler:
    distribution: str
    min_nmd: float = 0.001
    max_nmd: float = 0.1
    discrete_values: tuple = ()

    def __post_init__(self):
        if self.distribution in ("uniform", "log-uniform"):
            if self.min_nmd <= 0 or self.max_nmd <= 0:
                raise ValueError("target NMD bounds must be positive")
            if self.min_nmd >= self.max_nmd:
                raise ValueError("min_nmd must be less than max_nmd")
        elif self.distribution == "discrete":
            if not self.discrete_values:
                raise ValueError("a discrete sampler needs `discrete_values`")
        else:
            raise ValueError(f"unknown distribution {self.distribution!r}; expected "
                             "'uniform', 'log-uniform' or 'discrete'")

    def sample(self, np_rng: np.random.Generator) -> float:
        if self.distribution == "uniform":
            return float(np_rng.uniform(self.min_nmd, self.max_nmd))
        if self.distribution == "log-uniform":
            log_value = np_rng.uniform(np.log(self.min_nmd), np.log(self.max_nmd))
            return float(np.exp(log_value))
        return float(np_rng.choice(np.asarray(self.discrete_values)))

    def per_env(self, num_envs: int, np_rng: np.random.Generator) -> np.ndarray:
        return np.array([self.sample(np_rng) for _ in range(num_envs)])


def build_sampler(spec) -> TargetNMDSampler | None:
    if not spec:
        return None
    spec = dict(spec)
    distribution = spec.pop("distribution", None)
    if distribution is None:
        raise ValueError("target_nmd_sampler needs a `distribution`")
    values = tuple(spec.pop("discrete_values", ()) or ())
    sampler = TargetNMDSampler(distribution=distribution,
                               min_nmd=float(spec.pop("min_nmd", 0.001)),
                               max_nmd=float(spec.pop("max_nmd", 0.1)),
                               discrete_values=values)
    if spec:
        raise ValueError(f"target_nmd_sampler has unknown keys: {sorted(spec)}")
    return sampler

