from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..config import EnvConfig
from ..loading import load_scenario
from .registry import get_scenario


def _capabilities(agent_capabilities):
    return None if agent_capabilities is None else dict(agent_capabilities)


def _load_football_wrapper_module():
    from ..loading import load_module

    return load_module("football_wrapper",
                         Path(__file__).resolve().parent / "scenarios" / "football_wrapper.py")


def team_size(env_config: EnvConfig) -> int:
    return env_config.max_agents if env_config.mixed_agent_training else env_config.num_agents


def make_env(env_config: EnvConfig, agent_capabilities: Mapping[str, Any] | None,
             device: str = "cpu",
             curriculum: Mapping[str, Any] | None = None):
    spec = get_scenario(env_config.scenario)
    n_agents = team_size(env_config)

    from vmas.simulator.environment import Environment

    scenario = (None if spec.paper_name == "football"
                else load_scenario(spec.file_stem, spec.module_name).Scenario())

    if spec.paper_name == "navigation":
        if env_config.scenario_kwargs:
            raise ValueError(
                f"navigation takes no scenario_kwargs, got {sorted(env_config.scenario_kwargs)}"
            )
        return Environment(
            scenario=scenario,
            num_envs=env_config.num_envs,
            device=device,
            continuous_actions=env_config.continuous_actions,
            n_agents=n_agents,
            n_food=n_agents,
            penalise_by_time=env_config.penalise_by_time,
            share_reward=env_config.share_reward,
            distance_shaping_coef=env_config.distance_shaping_coef,
            agent_capabilities=_capabilities(agent_capabilities),
            fixed_food_positions=env_config.fixed_food_positions,
        )

    if spec.paper_name == "dispersion":
        kwargs = dict(env_config.scenario_kwargs)
        return Environment(
            scenario=scenario,
            num_envs=env_config.num_envs,
            device=device,
            continuous_actions=env_config.continuous_actions,
            fixed_n_food=kwargs.pop("fixed_n_food", None),
            n_agents=n_agents,
            n_food=n_agents,
            penalise_by_time=env_config.penalise_by_time,
            share_reward=env_config.share_reward,
            **kwargs,
        )

    if spec.paper_name == "reverse_transport":
        kwargs = dict(env_config.scenario_kwargs)
        kwargs.pop("scale_package_mass_with_agents", None)
        return Environment(
            scenario=scenario,
            num_envs=env_config.num_envs,
            device=device,
            continuous_actions=env_config.continuous_actions,
            n_agents=n_agents,
            package_width=kwargs.pop("package_width"),
            package_length=kwargs.pop("package_length"),
            package_mass=kwargs.pop("package_mass"),
            package_mass_range=list(kwargs.pop("package_mass_range")),
            agent_capabilities=_capabilities(agent_capabilities),
            **kwargs,
        )

    if spec.paper_name == "pressure_plate":
        kwargs = dict(env_config.scenario_kwargs)
        env = Environment(
            scenario=scenario,
            num_envs=env_config.num_envs,
            device=device,
            continuous_actions=env_config.continuous_actions,
            n_ground_robots=n_agents,
            share_reward=env_config.share_reward,
            agent_capabilities=_capabilities(agent_capabilities),
            eval_mode=False,
            **kwargs,
        )
        if not hasattr(env.scenario, "plate_reward"):
            env.scenario.plate_reward = kwargs.get("plate_reward", 0.1)
        return env

    if spec.paper_name == "wind_flocking":
        kwargs = dict(env_config.scenario_kwargs)
        return Environment(
            scenario=scenario,
            num_envs=env_config.num_envs,
            device=device,
            continuous_actions=env_config.continuous_actions,
            n_agents=n_agents,
            agent_radii=list(kwargs.pop("agent_radii")),
            **kwargs,
        )

    if spec.paper_name == "football":
        FootballWrapper = _load_football_wrapper_module().FootballWrapper

        kwargs = dict(env_config.scenario_kwargs)
        kwargs.pop("curriculum_stage1_episodes", None)
        kwargs.pop("curriculum_stage2_episodes", None)
        kwargs.update(curriculum or {})
        if "max_speed" not in kwargs and agent_capabilities is not None:
            speeds = (agent_capabilities.get("max_speed")
                      or agent_capabilities.get("speed") or [])
            if len(speeds):
                kwargs["max_speed"] = float(speeds[0])
        return FootballWrapper(
            num_blue_agents=n_agents,
            num_red_agents=kwargs.pop("num_red_agents"),
            num_envs=env_config.num_envs,
            device=device,
            continuous_actions=env_config.continuous_actions,
            ai_red_agents=kwargs.pop("ai_red_agents"),
            ai_blue_agents=kwargs.pop("ai_blue_agents"),
            physically_different=kwargs.pop("physically_different"),
            enable_shooting=kwargs.pop("enable_shooting"),
            dense_reward=kwargs.pop("dense_reward"),
            observe_teammates=kwargs.pop("observe_teammates"),
            observe_adversaries=kwargs.pop("observe_adversaries"),
            scenario_kwargs=kwargs or None,
        )

    raise NotImplementedError(
        f"environment {spec.paper_name!r} is not supported"
    )

