from __future__ import annotations

from typing import Any

import numpy as np

from ..config import EnvConfig
from .factory import team_size


def sample_team_size(env_config: EnvConfig, np_rng: np.random.Generator) -> int:
    if not env_config.mixed_agent_training:
        return env_config.num_agents
    return int(np_rng.integers(env_config.min_agents, env_config.max_agents + 1))


def sample_capabilities(env_config: EnvConfig, current_num_agents: int,
                        np_rng: np.random.Generator) -> dict[str, list[Any]]:
    if env_config.scenario == "dispersion":
        return _dispersion_capabilities(env_config, current_num_agents, np_rng)
    if env_config.scenario == "reverse_transport":
        return _reverse_transport_capabilities(env_config, current_num_agents, np_rng)
    if env_config.scenario == "pressure_plate":
        return _navigation_capabilities(env_config, current_num_agents, np_rng)
    return _navigation_capabilities(env_config, current_num_agents, np_rng)


def _reverse_transport_capabilities(env_config: EnvConfig, current_num_agents: int,
                                    np_rng: np.random.Generator) -> dict[str, list[Any]]:
    fixed = env_config.fixed_capabilities
    if env_config.mixed_agent_training:
        if env_config.use_fixed_capabilities:
            base = initial_capabilities(env_config)
            speeds = list(base["speed"])[:current_num_agents]
            forces = list(base["force_multiplier"])[:current_num_agents]
        elif env_config.homogeneous_capabilities:
            speeds = [np_rng.uniform(*env_config.speed_range)] * current_num_agents
            forces = [np_rng.uniform(*env_config.force_multiplier_range)] * current_num_agents
        else:
            speeds = np_rng.uniform(*env_config.speed_range, size=current_num_agents).tolist()
            forces = np_rng.uniform(*env_config.force_multiplier_range,
                                    size=current_num_agents).tolist()
        pad = team_size(env_config) - current_num_agents
        return {"speed": speeds + [1.0] * pad, "force_multiplier": forces + [1.0] * pad}

    if env_config.use_fixed_capabilities:
        base = initial_capabilities(env_config)
        return {"speed": list(base["speed"]),
                "force_multiplier": list(base["force_multiplier"])}
    return {
        "speed": np_rng.uniform(*env_config.speed_range, size=env_config.num_agents).tolist(),
        "force_multiplier": np_rng.uniform(*env_config.force_multiplier_range,
                                           size=env_config.num_agents).tolist(),
    }


def _dispersion_capabilities(env_config: EnvConfig, current_num_agents: int,
                             np_rng: np.random.Generator) -> dict[str, list[Any]]:
    if env_config.mixed_agent_training:
        if env_config.use_fixed_capabilities:
            speeds = [1.0] * current_num_agents
        elif env_config.homogeneous_capabilities:
            speeds = [np_rng.uniform(*env_config.max_speed_range)] * current_num_agents
        else:
            speeds = np_rng.uniform(*env_config.max_speed_range,
                                    size=current_num_agents).tolist()
        speeds += [1.0] * (team_size(env_config) - current_num_agents)
        return {"max_speed": speeds}

    if env_config.use_fixed_capabilities:
        return {"max_speed": list(initial_capabilities(env_config)["max_speed"])}
    if env_config.homogeneous_capabilities:
        base = np_rng.uniform(*env_config.max_speed_range)
        return {"max_speed": [base] * env_config.num_agents}
    return {"max_speed": np_rng.uniform(*env_config.max_speed_range,
                                        size=env_config.num_agents).tolist()}


def _navigation_capabilities(env_config: EnvConfig, current_num_agents: int,
                             np_rng: np.random.Generator) -> dict[str, list[Any]]:
    n_env_agents = team_size(env_config)

    if env_config.mixed_agent_training:
        if env_config.use_fixed_capabilities:
            speeds = [0.5] * current_num_agents
            lidar_ranges = [0.5] * current_num_agents
        elif env_config.homogeneous_capabilities:
            base_speed = np_rng.uniform(*env_config.speed_range)
            base_lidar = np_rng.uniform(*env_config.lidar_range_range)
            speeds = [base_speed] * current_num_agents
            lidar_ranges = [base_lidar] * current_num_agents
        else:
            speeds = np_rng.uniform(*env_config.speed_range, size=current_num_agents).tolist()
            lidar_ranges = np_rng.uniform(*env_config.lidar_range_range,
                                          size=current_num_agents).tolist()
        speeds += [1.0] * (n_env_agents - current_num_agents)
        lidar_ranges += [0.5] * (n_env_agents - current_num_agents)
        return {"speed": speeds, "lidar_range": lidar_ranges}

    if env_config.use_fixed_capabilities:
        base = initial_capabilities(env_config)
        return {"speed": list(base["speed"]), "lidar_range": list(base["lidar_range"])}

    return {
        "speed": np_rng.uniform(*env_config.speed_range, size=env_config.num_agents).tolist(),
        "lidar_range": np_rng.uniform(*env_config.lidar_range_range,
                                      size=env_config.num_agents).tolist(),
    }


def initial_capabilities(env_config: EnvConfig) -> dict[str, list[float]]:
    n = team_size(env_config)
    if env_config.scenario == "reverse_transport":
        fixed = env_config.fixed_capabilities
        if env_config.use_fixed_capabilities:
            speeds, forces = list(fixed.get("speed", [])), list(fixed.get("force_multiplier", []))
            if len(speeds) != n or len(forces) != n:
                return {"speed": [1.0] * n, "force_multiplier": [0.5] * n}
            return {"speed": speeds, "force_multiplier": forces}
        return {"speed": [1.0] * n, "force_multiplier": [0.5] * n}
    if env_config.scenario == "dispersion":
        if env_config.use_fixed_capabilities:
            speeds = list(env_config.fixed_capabilities.get("max_speed", []))
            if len(speeds) != n:
                values = [0.5, 0.75, 1.0, 1.25, 1.5]
                speeds = [values[i % len(values)] for i in range(n)]
            return {"max_speed": speeds}
        return {"max_speed": [1.0] * n}
    if env_config.use_fixed_capabilities:
        fixed = env_config.fixed_capabilities
        speeds, lidar = list(fixed.get("speed", [])), list(fixed.get("lidar_range", []))
        if len(speeds) != n or len(lidar) != n:
            speeds = [[0.5, 1.5][i % 2] for i in range(n)]
            lidar = [0.5] * n
        return {"speed": speeds, "lidar_range": lidar}
    return {"speed": [1.0] * n, "lidar_range": [0.5] * n}

