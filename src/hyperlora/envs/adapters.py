from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..diversity.nmd import (calculate_nmd, calculate_nmd_dispersion,
                             wind_flocking_adapter_positions)
from .events.base import NoEvents
from .events.navigation import NavigationEvents
from .events.pressure_plate import PressurePlateEvents
from .events.wind_flocking import WindFlockingEvents
from .tokens import (dispersion_inputs, football_inputs, navigation_inputs,
                     pressure_plate_env_context,
                     pressure_plate_inputs, reverse_transport_inputs,
                     wind_flocking_inputs)


@dataclass(frozen=True)
class EnvAdapter:
    build_tokens: Callable
    make_events: Callable
    nmd: Callable
    buffer_token_fields: tuple = ()
    apply_capabilities: Callable = None
    env_context: Callable = None


def _apply_capabilities(config, env, capabilities, current_num_agents):
    env.scenario.update_agent_capabilities(capabilities)


def _apply_capabilities_reverse_transport(config, env, capabilities, current_num_agents):
    env.scenario.update_agent_capabilities(capabilities)
    kwargs = config.env.scenario_kwargs
    if kwargs.get("scale_package_mass_with_agents", False):
        mass_per_agent = kwargs["package_mass"] / config.env.num_agents
        env.scenario.update_package_properties(package_mass=mass_per_agent * current_num_agents)


def _apply_capabilities_pressure_plate(config, env, capabilities, current_num_agents):
    return None


def _navigation_tokens(config, env, obs, obs_dim, num_envs, num_agents, current_num_agents,
                       target_nmd):
    return navigation_inputs(config, obs, obs_dim, current_num_agents, num_agents)


def _dispersion_tokens(config, env, obs, obs_dim, num_envs, num_agents, current_num_agents,
                       target_nmd):
    return dispersion_inputs(config, env, obs, num_envs, num_agents, target_nmd)


def _reverse_transport_tokens(config, env, obs, obs_dim, num_envs, num_agents,
                              current_num_agents, target_nmd):
    return reverse_transport_inputs(config, env, num_envs, num_agents, current_num_agents)


def _pressure_plate_tokens(config, env, obs, obs_dim, num_envs, num_agents,
                           current_num_agents, target_nmd):
    return pressure_plate_inputs(config, env, num_envs, num_agents, current_num_agents)


def _wind_flocking_tokens(config, env, obs, obs_dim, num_envs, num_agents, current_num_agents,
                          target_nmd):
    return wind_flocking_inputs(config, obs, num_envs, num_agents, target_nmd)


def _reverse_transport_env_context(config, env, num_envs, num_agents):
    import torch

    package = env.scenario.get_package_properties()
    row = torch.tensor([package["mass"], package["width"], package["length"]],
                       dtype=torch.float32)
    return row.unsqueeze(0).unsqueeze(0).expand(num_envs, num_agents, 3)


def _pressure_plate_env_context(config, env, num_envs, num_agents):
    return pressure_plate_env_context(config, env, num_envs, num_agents)


def _football_tokens(config, env, obs, obs_dim, num_envs, num_agents, current_num_agents,
                     target_nmd):
    return football_inputs(config, env, num_envs, num_agents)


def _apply_capabilities_football(config, env, capabilities, current_num_agents):
    return None


def _navigation_nmd(*, buffer, config, hn_params, policy_params, hypernetwork, policy,
                    num_agents, rng_key, trajectory_obs):
    return calculate_nmd(buffer, config.diversity.adapter_nmd_sample_size, hn_params,
                         policy_params, hypernetwork, policy, rng_key, trajectory_obs)


def _dispersion_nmd(*, buffer, config, hn_params, policy_params, hypernetwork, policy,
                    num_agents, rng_key, trajectory_obs):
    return calculate_nmd_dispersion(buffer, config.diversity.adapter_nmd_sample_size, hn_params,
                                    policy_params, hypernetwork, policy, num_agents, rng_key,
                                    trajectory_obs)


def _wind_flocking_nmd(*, buffer, config, hn_params, policy_params, hypernetwork, policy,
                       num_agents, rng_key, trajectory_obs):
    return calculate_nmd(
        buffer, config.diversity.adapter_nmd_sample_size, hn_params, policy_params, hypernetwork,
        policy, rng_key, trajectory_obs,
        agent_positions_fn=lambda n: wind_flocking_adapter_positions(n, num_agents))


ADAPTERS = {
    "navigation": EnvAdapter(
        build_tokens=_navigation_tokens,
        make_events=lambda config, num_envs, num_agents, obs_dim, episode, ema: NavigationEvents(
            num_envs, num_agents, config.model.adaptive_hypernetwork, obs_dim),
        nmd=_navigation_nmd,
        buffer_token_fields=("lidar",),
        apply_capabilities=_apply_capabilities,
    ),
    "dispersion": EnvAdapter(
        build_tokens=_dispersion_tokens,
        make_events=lambda config, num_envs, num_agents, obs_dim, episode, ema: NoEvents(
            num_envs, num_agents),
        nmd=_dispersion_nmd,
        buffer_token_fields=("context", "target_nmd"),
        apply_capabilities=_apply_capabilities,
    ),
    "reverse_transport": EnvAdapter(
        build_tokens=_reverse_transport_tokens,
        make_events=lambda config, num_envs, num_agents, obs_dim, episode, ema: NoEvents(
            num_envs, num_agents),
        nmd=_navigation_nmd,
        buffer_token_fields=("context",),
        apply_capabilities=_apply_capabilities_reverse_transport,
        env_context=_reverse_transport_env_context,
    ),
    "pressure_plate": EnvAdapter(
        build_tokens=_pressure_plate_tokens,
        make_events=lambda config, num_envs, num_agents, obs_dim, episode, ema: PressurePlateEvents(
            config, num_envs, num_agents),
        nmd=_navigation_nmd,
        buffer_token_fields=("env_context",),
        apply_capabilities=_apply_capabilities_pressure_plate,
        env_context=_pressure_plate_env_context,
    ),
    "football": EnvAdapter(
        build_tokens=_football_tokens,
        make_events=lambda config, num_envs, num_agents, obs_dim, episode, ema: NoEvents(
            num_envs, num_agents),
        nmd=_navigation_nmd,
        buffer_token_fields=("context",),
        apply_capabilities=_apply_capabilities_football,
    ),
    "wind_flocking": EnvAdapter(
        build_tokens=_wind_flocking_tokens,
        make_events=lambda config, num_envs, num_agents, obs_dim, episode, ema: WindFlockingEvents(
            config, num_envs, num_agents, episode, ema),
        nmd=_wind_flocking_nmd,
        buffer_token_fields=("agent_positions", "target_nmd"),
        apply_capabilities=_apply_capabilities,
    ),
}


def get_adapter(scenario: str) -> EnvAdapter:
    try:
        return ADAPTERS[scenario]
    except KeyError:
        raise NotImplementedError(
            f"environment {scenario!r} is not supported; supported: {sorted(ADAPTERS)}"
        ) from None

