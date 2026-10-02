from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("mappo", "vmas_wrapper_utils", "hypermarl_eval_utils",
            "mappo_ff_shared_weights_mlp_hypernets")


def _install_runtime_shims() -> None:
    import jax
    import orbax.checkpoint as ocp

    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map

    class _NoopCheckpointer:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            pass

    checkpointer = types.ModuleType("orbax.checkpoint.checkpointer")
    checkpointer.Checkpointer = _NoopCheckpointer
    sys.modules["orbax.checkpoint.checkpointer"] = checkpointer
    ocp.checkpointer = checkpointer
    handler = types.ModuleType("orbax.checkpoint.pytree_checkpoint_handler")
    handler.PyTreeCheckpointHandler = getattr(ocp, "PyTreeCheckpointHandler", object)
    sys.modules["orbax.checkpoint.pytree_checkpoint_handler"] = handler


def _make_vmas_env(scenario_name, num_agents, num_envs, device="cpu", continuous_actions=True,
                   penalise_by_time=False, share_reward=False, distance_shaping_coef=1.0,
                   agent_capabilities=None, fixed_food_positions=False, fixed_n_food=None,
                   use_original_vmas_env=False, **kwargs):
    from ...config import EnvConfig
    from ...envs.factory import make_env as build
    from ...envs.registry import CODE_NAME_TO_PAPER

    if scenario_name not in CODE_NAME_TO_PAPER:
        raise KeyError(
            f"unknown scenario {scenario_name!r}; expected one of {sorted(CODE_NAME_TO_PAPER)}")

    FORWARDED = {
        "navigation": (),
        "dispersion": ("fixed_n_food",),
        "reverse_transport": ("package_width", "package_length", "package_mass",
                              "package_mass_range"),
        "pressure_plate": ("x_semidim", "y_semidim", "plate_radius", "plate_margin", "door_size",
                           "goal_radius", "with_drone", "use_global_obs", "plate_reward",
                           "goal_reward", "time_penalty", "reward_type", "training_spawn_side"),
        "football": ("num_red_agents", "ai_red_agents", "ai_blue_agents", "physically_different",
                     "enable_shooting", "dense_reward", "observe_teammates", "observe_adversaries",
                     "disable_ai_red", "pos_shaping_factor_ball_goal",
                     "pos_shaping_factor_agent_ball", "max_speed"),
        "wind_flocking": (),
    }
    DEFAULTS = {
        "reverse_transport": {"package_width": 0.6, "package_length": 0.6, "package_mass": 50,
                              "package_mass_range": [1, 100]},
        "football": {"num_red_agents": num_agents, "ai_red_agents": True, "ai_blue_agents": False,
                     "physically_different": False, "enable_shooting": False, "dense_reward": True,
                     "observe_teammates": True, "observe_adversaries": True},
    }
    paper_name = CODE_NAME_TO_PAPER[scenario_name]
    allowed = FORWARDED[paper_name]
    scenario_kwargs = dict(DEFAULTS.get(paper_name, {}))
    scenario_kwargs.update({k: v for k, v in kwargs.items() if k in allowed})
    if fixed_n_food is not None and "fixed_n_food" in allowed:
        scenario_kwargs["fixed_n_food"] = fixed_n_food

    if paper_name == "pressure_plate":
        num_agents = int(kwargs.get("n_ground_robots", 3))

    env_config = EnvConfig(
        scenario=paper_name,
        num_agents=num_agents, max_agents=num_agents, num_envs=num_envs,
        continuous_actions=continuous_actions, penalise_by_time=penalise_by_time,
        share_reward=share_reward, distance_shaping_coef=distance_shaping_coef,
        mixed_agent_training=False, min_agents=num_agents,
        use_fixed_capabilities=agent_capabilities is not None,
        homogeneous_capabilities=True,
        fixed_capabilities=dict(agent_capabilities or {}),
        fixed_food_positions=fixed_food_positions,
        speed_range=(0.5, 1.5), lidar_range_range=(0.3, 0.7), max_speed_range=(0.5, 1.5),
        force_multiplier_range=(0.5, 1.5), scenario_kwargs=scenario_kwargs,
    )
    return build(env_config, agent_capabilities, device=device)


def install() -> types.ModuleType:
    _install_runtime_shims()

    if "env_setup" not in sys.modules:
        env_setup = types.ModuleType("env_setup")
        env_setup.make_vmas_env = _make_vmas_env
        sys.modules["env_setup"] = env_setup

    from ...loading import load_module
    from ...envs.factory import _load_football_wrapper_module

    _load_football_wrapper_module()
    for name in MODULES:
        load_module(name, HERE / f"{name}.py")
    return sys.modules["mappo_ff_shared_weights_mlp_hypernets"]

