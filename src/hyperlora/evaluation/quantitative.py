from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..config import Config
from ..envs.tokens import capability_context


def eval_capabilities(config: Config, num_agents: int) -> Mapping[str, list] | None:
    env = config.env
    if env.scenario == "dispersion":
        speeds = config.evaluation.eval_max_speeds or (1.0,)
        return {"max_speed": [float(speeds[0])] * num_agents}
    if config.evaluation.match_training_capabilities:
        if not env.use_fixed_capabilities:
            raise ValueError("evaluation.match_training_capabilities needs "
                             "env.use_fixed_capabilities: the training capabilities are otherwise "
                             "drawn per episode and there is no single condition to match")
        from ..envs.capabilities import sample_capabilities
        return sample_capabilities(env, num_agents, np.random.default_rng(0))
    if env.mixed_agent_training:
        return {"speed": [0.5] * num_agents, "lidar_range": [0.5] * num_agents}
    if env.use_fixed_capabilities:
        return {k: list(v) for k, v in env.fixed_capabilities.items()}
    return None


def _match_training_package_mass(config: Config, env, num_agents: int) -> None:
    kwargs = config.env.scenario_kwargs
    scenario = getattr(env, "scenario", None)
    if not kwargs.get("scale_package_mass_with_agents", False):
        return
    if scenario is None or not hasattr(scenario, "update_package_properties"):
        return
    mass_per_agent = kwargs["package_mass"] / config.env.num_agents
    scenario.update_package_properties(package_mass=mass_per_agent * num_agents)


def _diversity_scaling(config: Config, nmd_ema: float | None) -> float:
    if config.method != "ours" or not config.diversity.use_diversity_control:
        return 1.0
    diversity = config.diversity
    scaling = float(np.sqrt(diversity.target_nmd / max(nmd_ema, diversity.min_nmd_floor)))
    return float(np.clip(scaling, 0.001, np.sqrt(diversity.max_diversity_scaling)))


def _agent_capabilities(config: Config, fixed: Mapping[str, list] | None, num_agents: int) -> dict:
    scenario = config.env.scenario
    if fixed is None:
        if scenario == "dispersion":
            return {"max_speed": [1.0 + 0.5 * (i % 2) for i in range(num_agents)]}
        if scenario == "reverse_transport":
            return {"speed": [0.5] * num_agents,
                    "force_multiplier": [1.0 + 0.5 * (i % 2) for i in range(num_agents)]}
        return {"speed": [[0.5, 1.5][i % 2] for i in range(num_agents)],
                "lidar_range": [0.5] * num_agents}

    speeds = list(fixed.get("speed", []))
    lidar = list(fixed.get("lidar_range", []))
    forces = fixed.get("force_multiplier")
    max_speeds = fixed.get("max_speed")

    if scenario == "reverse_transport":
        needs_fallback = len(speeds) < num_agents
    elif scenario == "dispersion":
        needs_fallback = max_speeds is None or len(max_speeds) < num_agents
    else:
        needs_fallback = len(speeds) < num_agents or len(lidar) < num_agents
    if needs_fallback:
        speeds = [[0.5, 1.5][i % 2] for i in range(num_agents)]
        lidar = [0.5] * num_agents
    if forces is not None and len(forces) < num_agents:
        values = list(forces) or [1.0, 1.0]
        forces = [values[i % len(values)] for i in range(num_agents)]
    if max_speeds is not None and len(max_speeds) < num_agents:
        values = list(max_speeds) or [1.0]
        max_speeds = [values[i % len(values)] for i in range(num_agents)]

    out: dict = {}
    if scenario == "reverse_transport":
        out["speed"] = list(speeds)[:num_agents]
        if forces is not None:
            out["force_multiplier"] = list(forces)[:num_agents]
    elif scenario != "dispersion":
        out["speed"] = list(speeds)[:num_agents]
        out["lidar_range"] = list(lidar)[:num_agents]
    if max_speeds is not None:
        out["max_speed"] = list(max_speeds)[:num_agents]
    return out


def _buffer_entries(tokens, fields: tuple, num_envs: int, num_agents: int) -> list:
    entries = []
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            entry = {"task": None, "context": None, "lidar": None, "target_nmd": None,
                     "query_type": "initial", "hidden_state": None, "agent_idx": agent_idx}
            for name in fields:
                value = getattr(tokens, name)
                entry[name] = None if value is None else value[env_idx][agent_idx]
            entries.append(entry)
    return entries


def _is_football(env) -> bool:
    return hasattr(env, "num_blue_agents") and hasattr(env, "num_red_agents")


def _is_dispersion_like(env) -> bool:
    scenario = getattr(env, "scenario", None)
    return scenario is not None and "dispersion" in type(scenario).__module__.lower()


def _update_eval_ema(config: Config, nmd_ema: float, measured: float) -> float:
    coef = config.diversity.nmd_moving_average_coef
    return coef * nmd_ema + (1.0 - coef) * measured


def run_quantitative_evaluation(*, config: Config, env, models, adapter, num_agents: int,
                                obs_dim: int, actions_fn, nmd_ema: float | None = None,
                                tokens_fn=None, num_eval_episodes: int | None = None,
                                max_eval_steps: int | None = None, after_step_fn=None) -> dict:
    evaluation = config.evaluation
    num_eval_episodes = num_eval_episodes or evaluation.num_eval_episodes
    max_eval_steps = max_eval_steps or evaluation.max_eval_steps
    num_envs = config.env.num_envs

    fixed = eval_capabilities(config, num_agents)
    if evaluation.match_training_capabilities:
        _match_training_package_mass(config, env, num_agents)
    scaling = _diversity_scaling(config, nmd_ema)

    is_football, is_dispersion_like = _is_football(env), _is_dispersion_like(env)
    total_rewards: list = []
    episode_lengths: list = []
    completed_episodes = 0
    components: dict[str, list] = {"food_collection": [], "shaping": [], "time_penalty": []}
    agents_at_goal: list = []
    total_food_available = 0
    total_food_collected = 0
    football_blue_goals = football_red_goals = 0
    football_wins = football_draws = football_losses = 0

    measures_nmd = (config.method == "ours" and config.diversity.use_diversity_control
                    and config.diversity.use_adapter_nmd
                    and models.hn_state is not None)

    for episode in range(num_eval_episodes):
        buffer: list = []
        trajectory_obs: list = []
        capabilities = _agent_capabilities(config, fixed, num_agents)
        scenario = getattr(env, "scenario", None)
        if scenario is not None and hasattr(scenario, "update_agent_capabilities"):
            scenario.update_agent_capabilities(capabilities)

        obs = env.reset()
        state = tokens_fn(env, obs, scaling) if tokens_fn is not None else None
        if measures_nmd and state is not None and "tokens" in state:
            buffer.extend(_buffer_entries(state["tokens"], adapter.buffer_token_fields,
                                          num_envs, num_agents))

        batch_rewards = np.zeros(num_envs)
        batch_lengths = np.zeros(num_envs)
        active_mask = np.ones(num_envs, dtype=bool)
        episode_blue = np.zeros(num_envs, dtype=int)
        episode_red = np.zeros(num_envs, dtype=int)

        food_eaten = None
        if is_dispersion_like:
            food_eaten = [food.eaten.clone() for food in env.world.landmarks]
            total_food_available += num_envs * len(env.world.landmarks)
        episode_food = 0

        for _step in range(max_eval_steps):
            obs_transposed = torch.stack(obs, dim=0).transpose(0, 1)
            jax_obs = jnp.asarray(
                obs_transposed.reshape(num_envs * num_agents, -1).cpu().numpy())

            if measures_nmd:
                trajectory_obs.append(jax_obs)
            actions, state = actions_fn(jax_obs, state)

            actions_flat = np.nan_to_num(np.asarray(actions), nan=0.0, posinf=1.0, neginf=-1.0)
            actions_flat = np.clip(actions_flat, -1.0, 1.0)
            actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
            torch_actions = [torch.tensor(actions_reshaped[:, i, :], dtype=torch.float32)
                             for i in range(num_agents)]
            next_obs, rewards, dones, info = env.step(torch_actions)
            if after_step_fn is not None and state is not None:
                state = after_step_fn(env, state)

            if is_dispersion_like:
                for index, food in enumerate(env.world.landmarks):
                    newly_eaten = food.eaten & ~food_eaten[index]
                    episode_food += int(newly_eaten.sum().item())
                    food_eaten[index] = food.eaten.clone()

            first_info = (info[0] if isinstance(info, list) else info) if info else None
            if isinstance(first_info, dict):
                for key, name in (("reward_food_collection", "food_collection"),
                                  ("reward_shaping", "shaping"),
                                  ("reward_time_penalty", "time_penalty")):
                    if key in first_info:
                        value = first_info[key]
                        components[name].append(value.item() if hasattr(value, "item") else value)
                if "percentage_agents_at_goal" in first_info:
                    value = first_info["percentage_agents_at_goal"]
                    agents_at_goal.extend(
                        value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value))
                if is_football and "sparse_reward" in first_info:
                    sparse = first_info["sparse_reward"]
                    sparse = sparse.cpu().numpy() if hasattr(sparse, "cpu") else np.asarray(sparse)
                    blue, red = (sparse > 50).astype(int), (sparse < -50).astype(int)
                    football_blue_goals += int(blue.sum())
                    football_red_goals += int(red.sum())
                    episode_blue += blue
                    episode_red += red

            batch_lengths[active_mask] += 1
            step_rewards = torch.stack(rewards, dim=0).transpose(0, 1).cpu().numpy().mean(axis=1)
            batch_rewards += step_rewards

            dones_np = dones.cpu().numpy()
            newly_done = dones_np & active_mask
            if is_football:
                completed_episodes += int(((batch_rewards > 0) & newly_done).sum())
            else:
                completed_episodes += int(newly_done.sum())
            active_mask = active_mask & (~dones_np)

            obs = next_obs
            if not active_mask.any():
                break

        if is_dispersion_like:
            total_food_collected += episode_food
        if is_football:
            for index in range(num_envs):
                if episode_blue[index] > episode_red[index]:
                    football_wins += 1
                elif episode_blue[index] == episode_red[index]:
                    football_draws += 1
                else:
                    football_losses += 1

        if measures_nmd and buffer:
            measured = adapter.nmd(
                buffer=buffer, config=config, hn_params=models.hn_state.params,
                policy_params=models.policy_state.params, hypernetwork=models.hypernetwork,
                policy=models.policy, num_agents=num_agents,
                rng_key=jax.random.PRNGKey(episode * 1000 + 42),
                trajectory_obs=trajectory_obs or None)
            if np.isfinite(measured):
                nmd_ema = _update_eval_ema(config, nmd_ema, measured)
            scaling = _diversity_scaling(config, nmd_ema)

        total_rewards.extend(batch_rewards)
        episode_lengths.extend(batch_lengths)

    total_env_episodes = num_eval_episodes * num_envs
    metrics: dict[str, Any] = {
        "completion_rate": (completed_episodes / total_env_episodes) * 100.0,
        "completed_episodes": completed_episodes,
        "total_episodes": total_env_episodes,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
    }
    decomposition = {name: float(np.mean(values)) for name, values in components.items() if values}
    if decomposition:
        metrics["reward_decomposition"] = decomposition
    if is_football:
        total_goals = football_blue_goals + football_red_goals
        counted = max(football_wins + football_draws + football_losses, 1)
        metrics.update({
            "football_blue_goals": football_blue_goals,
            "football_red_goals": football_red_goals,
            "football_total_goals": total_goals,
            "football_win_rate": (football_blue_goals / max(total_goals, 1)) * 100.0,
            "football_wins": football_wins,
            "football_draws": football_draws,
            "football_losses": football_losses,
            "football_episode_win_rate": (football_wins / counted) * 100.0,
            "football_episode_draw_rate": (football_draws / counted) * 100.0,
        })
    if total_food_available > 0:
        key = "success_rate" if config.env.scenario == "navigation" else "food_found_percentage"
        metrics[key] = (total_food_collected / total_food_available) * 100.0
    if agents_at_goal:
        metrics["avg_agents_at_goal_percentage"] = float(np.mean(agents_at_goal)) * 100.0
        metrics["std_agents_at_goal_percentage"] = float(np.std(agents_at_goal)) * 100.0
    return metrics

