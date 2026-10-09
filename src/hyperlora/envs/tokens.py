from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import torch

from ..config import Config


@dataclass(frozen=True)
class HypernetworkInputs:
    task: jnp.ndarray | None = None
    context: jnp.ndarray | None = None
    lidar: jnp.ndarray | None = None
    food_positions: jnp.ndarray | None = None
    agent_positions: jnp.ndarray | None = None
    target_nmd: jnp.ndarray | None = None
    env_context: jnp.ndarray | None = None
    mask: jnp.ndarray | None = None

    FIELDS = ("task", "context", "lidar", "food_positions", "agent_positions", "target_nmd",
              "env_context")

    def as_args(self):
        return (self.task, self.context, self.lidar, self.food_positions, self.agent_positions,
                self.target_nmd, self.env_context, self.mask)

    def flatten_rows(self, batch_size: int) -> dict:
        rows = {}
        for name in self.FIELDS:
            value = getattr(self, name)
            rows[name] = None if value is None else value.reshape(batch_size, -1)
        return rows

    def with_rows(self, indices, replacement: "HypernetworkInputs") -> "HypernetworkInputs":
        updated = {}
        for name in self.FIELDS:
            current, new = getattr(self, name), getattr(replacement, name)
            if current is None or new is None:
                updated[name] = current
                continue
            num_envs, num_agents = current.shape[0], current.shape[1]
            flat = current.reshape(num_envs * num_agents, -1)
            flat = flat.at[indices].set(new.reshape(len(indices), -1))
            updated[name] = flat.reshape(num_envs, num_agents, -1)
        return HypernetworkInputs(mask=self.mask, **updated)


def create_attention_mask(current_num_agents: int, max_agents: int):
    if current_num_agents == max_agents:
        return None
    mask = jnp.ones((max_agents, max_agents), dtype=bool)
    mask = mask.at[current_num_agents:, :].set(False)
    mask = mask.at[:, current_num_agents:].set(False)
    return mask[None, None, :, :]


def lidar_dim(obs_dim: int) -> int:
    return obs_dim - 8


def extract_lidar(obs_list: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.stack([agent_obs[:, -dim:] for agent_obs in obs_list], dim=0).transpose(0, 1)


def lidar_to_jax(lidar: torch.Tensor) -> jnp.ndarray:
    array = lidar.detach().cpu().numpy() if lidar.requires_grad else lidar.cpu().numpy()
    array = np.nan_to_num(array, posinf=1.0, neginf=0.0)
    array = np.clip(array, -1.0, 1.0)
    return jnp.asarray(array)


def generate_positional_encoding(num_agents: int, encoding_dim: int) -> torch.Tensor:
    positions = torch.arange(num_agents, dtype=torch.float32).unsqueeze(1)
    dim_indices = torch.arange(encoding_dim, dtype=torch.float32)
    div_term = torch.exp(dim_indices * -(np.log(10000.0) / encoding_dim))
    encoding = torch.zeros(num_agents, encoding_dim)
    encoding[:, 0::2] = torch.sin(positions * div_term[0::2])
    encoding[:, 1::2] = torch.cos(positions * div_term[1::2])
    return encoding


def target_token(target_nmd, num_envs: int, env_num_agents: int, dim: int) -> torch.Tensor:
    values = torch.as_tensor(target_nmd, dtype=torch.float32).reshape(-1)
    if values.numel() == 1:
        return torch.full((num_envs, env_num_agents, dim), float(values[0]), dtype=torch.float32)
    if values.numel() != num_envs:
        raise ValueError(f"target NMD has {values.numel()} values for {num_envs} environments")
    return values[:, None, None].expand(num_envs, env_num_agents, dim).contiguous()


def extract_agent_positions(obs_list: list[torch.Tensor]) -> torch.Tensor:
    positions = torch.stack([agent_obs[:, :2] for agent_obs in obs_list], dim=0).transpose(0, 1)
    return positions - positions.mean(dim=1).unsqueeze(1)


def dispersion_inputs(config: Config, env, obs_list: list[torch.Tensor], num_envs: int,
                      env_num_agents: int, target_nmd: float) -> HypernetworkInputs:
    model = config.model
    if not (model.use_capability_context and model.use_positional_context):
        raise NotImplementedError("Dispersion expects capability + positional context ")

    capabilities = torch.stack(
        [torch.tensor([float(getattr(env.agents[i], "_max_speed"))], dtype=torch.float32)
         for i in range(env_num_agents)], dim=0)
    context = capabilities.unsqueeze(0).expand(num_envs, -1, -1)
    encoding = generate_positional_encoding(env_num_agents, model.positional_encoding_dim)
    context = torch.cat([context, encoding.unsqueeze(0).expand(num_envs, -1, -1)], dim=-1)

    target = target_token(target_nmd, num_envs, env_num_agents, model.target_nmd_dim)

    return HypernetworkInputs(
        context=jnp.asarray(context.cpu().numpy()),
        agent_positions=jnp.asarray(extract_agent_positions(obs_list).cpu().numpy()),
        target_nmd=jnp.asarray(target.cpu().numpy()),
        mask=None,
    )


def reverse_transport_inputs(config: Config, env, num_envs: int, env_num_agents: int,
                             current_num_agents: int) -> HypernetworkInputs:
    if not config.model.use_capability_context:
        raise NotImplementedError("Reverse Transport expects the capability context ")

    capabilities = torch.stack([
        torch.tensor([float(getattr(env.agents[i], "_max_speed")),
                      float(getattr(env.agents[i], "_force_multiplier"))], dtype=torch.float32)
        for i in range(env_num_agents)], dim=0)
    context = capabilities.unsqueeze(0).expand(num_envs, -1, -1)

    package = env.scenario.get_package_properties()
    env_context = torch.tensor([package["mass"], package["width"], package["length"]],
                               dtype=torch.float32)
    env_context = env_context.unsqueeze(0).unsqueeze(0).expand(num_envs, env_num_agents, 3)

    mask = (create_attention_mask(current_num_agents, env_num_agents)
            if current_num_agents < env_num_agents else None)
    return HypernetworkInputs(
        context=jnp.asarray(context.cpu().numpy()),
        env_context=jnp.asarray(env_context.cpu().numpy()),
        mask=mask,
    )


def navigation_inputs(config: Config, obs_list: list[torch.Tensor], obs_dim: int,
                      current_num_agents: int, env_num_agents: int) -> HypernetworkInputs:
    model = config.model
    if model.use_capability_context or model.task_embed_dim or model.target_nmd_dim:
        raise NotImplementedError(
            "the Navigation token is LiDAR-only"
        )
    if not model.use_lidar_context:
        raise ValueError("Navigation requires use_lidar_context: true")

    lidar = lidar_to_jax(extract_lidar(obs_list, lidar_dim(obs_dim)))
    mask = (create_attention_mask(current_num_agents, env_num_agents)
            if current_num_agents < env_num_agents else None)
    return HypernetworkInputs(lidar=lidar, mask=mask)


PRESSURE_PLATE_ENV_CONTEXT_FLAGS = ("plate_positions", "door_state", "goal_position", "agent_ids")


def pressure_plate_env_context_dim(flags, max_agents: int) -> int:
    if not flags:
        return 0
    if sorted(flags) != sorted(PRESSURE_PLATE_ENV_CONTEXT_FLAGS):
        raise ValueError(
            f"model.env_context must give all of {list(PRESSURE_PLATE_ENV_CONTEXT_FLAGS)} "
            f"or be empty, got {sorted(flags)}")
    return (4 * flags["plate_positions"] + 1 * flags["door_state"]
            + 2 * flags["goal_position"] + max_agents * flags["agent_ids"])


def pressure_plate_env_context(config: Config, env, num_envs: int,
                               env_num_agents: int) -> torch.Tensor:
    flags = config.model.env_context
    robots = sorted([a for a in env.agents if "ground_robot" in a.name], key=lambda a: a.name)
    agent_pos = torch.stack([a.state.pos[:, :2] for a in robots], dim=1)

    parts = []
    if flags["plate_positions"]:
        left = env.scenario.plate_left.state.pos[:, :2]
        right = env.scenario.plate_right.state.pos[:, :2]
        parts.extend([left.unsqueeze(1) - agent_pos, right.unsqueeze(1) - agent_pos])
    if flags["door_state"]:
        parts.append(env.scenario.door_open.float()[:, None, None].expand(
            num_envs, env_num_agents, 1))
    if flags["goal_position"]:
        parts.append(env.scenario.goal.state.pos[:, :2].unsqueeze(1) - agent_pos)
    if flags["agent_ids"]:
        one_hot = torch.zeros(num_envs, env_num_agents, config.env.max_agents)
        for i in range(env_num_agents):
            one_hot[:, i, i] = 1.0
        parts.append(one_hot)
    return torch.cat(parts, dim=-1)


def pressure_plate_inputs(config: Config, env, num_envs: int, env_num_agents: int,
                          current_num_agents: int) -> HypernetworkInputs:
    if not config.model.env_context:
        raise ValueError(
            "Pressure Plate with an empty model.env_context would leave the hypernetwork with no "
            "token at all")

    env_context = pressure_plate_env_context(config, env, num_envs, env_num_agents)
    mask = (create_attention_mask(current_num_agents, env_num_agents)
            if current_num_agents < env_num_agents else None)
    return HypernetworkInputs(env_context=jnp.asarray(env_context.cpu().numpy()), mask=mask)


def wind_flocking_inputs(config: Config, obs_list: list[torch.Tensor], num_envs: int,
                         env_num_agents: int, target_nmd: float) -> HypernetworkInputs:
    model = config.model
    if not model.use_agent_position_context or model.agent_position_dim != 2:
        raise NotImplementedError("Wind Flocking expects a 2-D agent-position context ")
    if model.target_nmd_dim != 1:
        raise NotImplementedError("Wind Flocking expects a 1-D target-NMD token ")

    positions = extract_agent_positions(obs_list)
    target = target_token(target_nmd, num_envs, env_num_agents, model.target_nmd_dim)
    return HypernetworkInputs(
        agent_positions=jnp.asarray(positions.cpu().numpy()),
        target_nmd=jnp.asarray(target.cpu().numpy()),
        mask=None,
    )


def football_inputs(config: Config, env, num_envs: int, env_num_agents: int) -> HypernetworkInputs:
    if not config.model.use_capability_context:
        raise NotImplementedError("Football expects the capability context ")

    capabilities = torch.from_numpy(env.get_capability_vectors(normalize=True))
    capabilities = capabilities.unsqueeze(0).expand(num_envs, -1, -1)
    positions = env.get_initial_positions(normalize=True)
    context = torch.cat([capabilities, positions], dim=-1)
    return HypernetworkInputs(context=jnp.asarray(context.cpu().numpy()), mask=None)


def capability_context(config: Config, env, num_envs: int, env_num_agents: int) -> torch.Tensor:
    model = config.model
    scenario = config.env.scenario

    if not model.use_capability_context or scenario == "pressure_plate":
        return torch.zeros(num_envs, env_num_agents, 0, dtype=torch.float32)

    if scenario == "football":
        capabilities = torch.from_numpy(env.get_capability_vectors(normalize=True))
        capabilities = capabilities.unsqueeze(0).expand(num_envs, -1, -1)
        return torch.cat([capabilities, env.get_initial_positions(normalize=True)], dim=-1)

    if scenario == "reverse_transport":
        rows = [[float(getattr(env.agents[i], "_max_speed")),
                 float(getattr(env.agents[i], "_force_multiplier"))]
                for i in range(env_num_agents)]
    elif scenario == "dispersion":
        rows = [[float(getattr(env.agents[i], "_max_speed"))] for i in range(env_num_agents)]
    else:
        rows = [[float(getattr(env.agents[i], "_max_speed")),
                 float(getattr(env.agents[i], "_obs_range"))] for i in range(env_num_agents)]

    context = torch.tensor(rows, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1, -1)

    if scenario == "dispersion":
        if model.use_positional_context:
            encoding = generate_positional_encoding(env_num_agents, model.positional_encoding_dim)
            context = torch.cat([context, encoding.unsqueeze(0).expand(num_envs, -1, -1)], dim=-1)
        elif model.use_onehot_context:
            one_hot = torch.nn.functional.one_hot(torch.arange(env_num_agents),
                                                  num_classes=config.env.max_agents).float()
            context = torch.cat([context, one_hot.unsqueeze(0).expand(num_envs, -1, -1)], dim=-1)
    return context

