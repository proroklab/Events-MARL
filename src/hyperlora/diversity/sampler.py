from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetSNDSampler:
    distribution: str
    min_snd: float = 0.001
    max_snd: float = 0.1
    discrete_values: tuple = ()

    def __post_init__(self):
        if self.distribution in ("uniform", "log-uniform"):
            if self.min_snd <= 0 or self.max_snd <= 0:
                raise ValueError("target NMD bounds must be positive")
            if self.min_snd >= self.max_snd:
                raise ValueError("min_snd must be less than max_snd")
        elif self.distribution == "discrete":
            if not self.discrete_values:
                raise ValueError("a discrete sampler needs `discrete_values`")
        else:
            raise ValueError(f"unknown distribution {self.distribution!r}; expected "
                             "'uniform', 'log-uniform' or 'discrete'")

    def sample(self, np_rng: np.random.Generator) -> float:
        if self.distribution == "uniform":
            return float(np_rng.uniform(self.min_snd, self.max_snd))
        if self.distribution == "log-uniform":
            log_value = np_rng.uniform(np.log(self.min_snd), np.log(self.max_snd))
            return float(np.exp(log_value))
        return float(np_rng.choice(np.asarray(self.discrete_values)))

    def per_env(self, num_envs: int, np_rng: np.random.Generator) -> np.ndarray:
        return np.array([self.sample(np_rng) for _ in range(num_envs)])


def build_sampler(spec) -> TargetSNDSampler | None:
    if not spec:
        return None
    spec = dict(spec)
    distribution = spec.pop("distribution", None)
    if distribution is None:
        raise ValueError("target_snd_sampler needs a `distribution`")
    values = tuple(spec.pop("discrete_values", ()) or ())
    sampler = TargetSNDSampler(distribution=distribution,
                               min_snd=float(spec.pop("min_snd", 0.001)),
                               max_snd=float(spec.pop("max_snd", 0.1)),
                               discrete_values=values)
    if spec:
        raise ValueError(f"target_snd_sampler has unknown keys: {sorted(spec)}")
    return sampler

