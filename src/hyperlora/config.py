from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

METHODS = ("ours", "ps", "dico", "cash")


@dataclass(frozen=True)
class EnvConfig:
    scenario: str
    num_agents: int
    max_agents: int
    num_envs: int
    continuous_actions: bool
    penalise_by_time: bool
    share_reward: bool
    distance_shaping_coef: float
    mixed_agent_training: bool
    min_agents: int
    use_fixed_capabilities: bool
    homogeneous_capabilities: bool
    fixed_capabilities: Mapping[str, list]
    fixed_food_positions: bool
    speed_range: tuple
    lidar_range_range: tuple
    max_speed_range: tuple
    force_multiplier_range: tuple
    scenario_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    policy_hidden_dims: tuple
    critic_hidden_dim: int
    critic_num_layers: int
    use_shared_critic: bool
    log_std_max: float
    log_std_min: float
    min_std: float
    lora_rank: int
    lora_mode: str
    lora_scaling_factor: float
    transformer_dim: int
    transformer_heads: int
    transformer_layers: int
    use_cross_agent_attention: bool
    task_embed_dim: int
    target_nmd_dim: int | None  # Ours only; absent (None) for the other methods
    use_capability_context: bool
    use_lidar_context: bool
    use_positional_context: bool
    positional_encoding_dim: int
    use_onehot_context: bool
    use_agent_position_context: bool
    agent_position_dim: int
    env_context: Mapping[str, bool]
    adaptive_hypernetwork: bool
    gru_hidden_dim: int | None
    fc_dim_size: int | None
    cash_hyper_hidden_dim: int | None
    cash_hyper_layers: int | None
    cash_decoder_hidden_dim: int | None
    cash_use_two_layer_decoder: bool | None


@dataclass(frozen=True)
class PPOConfig:
    num_episodes: int
    rollout_steps: int
    ppo_epochs: int
    ppo_clip_epsilon: float
    entropy_coef: float
    entropy_decay: float
    min_entropy_coef: float
    value_loss_coef: float
    gamma: float
    gae_lambda: float
    reward_scale: float
    advantage_clip_range: float
    normalize_returns: bool
    clip_value_loss: bool
    clip_global_state: bool
    clip_global_state_min: float
    clip_global_state_max: float
    use_minibatches: bool
    num_minibatches: int
    shuffle_minibatches: bool
    max_grad_norm: float
    hypernetwork_grad_norm: float
    critic_grad_norm: float
    adapter_regularization_coef: float


@dataclass(frozen=True)
class NMDConfig:
    """Ours: diversity measured as NMD between hypernetwork behaviours."""
    use_diversity_control: bool
    use_adapter_nmd: bool
    target_nmd: float
    nmd_moving_average_coef: float
    max_diversity_scaling: float
    min_nmd_floor: float
    adapter_nmd_sample_size: int
    target_nmd_list: tuple
    target_nmd_change_interval: int
    target_nmd_mode: str
    target_nmd_sampler: dict


@dataclass(frozen=True)
class SNDConfig:
    """DiCo: diversity measured as SND between agents (Bettini et al.)."""
    use_diversity_control: bool
    target_snd: float
    snd_moving_average_coef: float
    max_diversity_scaling: float
    min_snd_floor: float
    num_snd_samples: int


# The `diversity` section each method takes; PS and CASH have none.
DIVERSITY_SECTIONS = {"ours": NMDConfig, "dico": SNDConfig, "ps": None, "cash": None}

# Fields only some methods take; for every other method the key must be absent and reads as None.
_METHOD_ONLY_FIELDS = {("model", "target_nmd_dim"): ("ours",)}


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    hn_learning_rate: float
    critic_learning_rate: float
    beta1: float
    beta2: float
    eps: float


@dataclass(frozen=True)
class EvalConfig:
    eval_interval: int
    eval_start_episode: int
    num_eval_episodes: int
    max_eval_steps: int
    eval_max_speeds: tuple

    match_training_capabilities: bool


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str
    checkpoint_dir: str
    save_interval: int
    experiment_name: str


@dataclass(frozen=True)
class Config:
    method: str
    seed: int
    jax_threefry_partitionable: bool
    env: EnvConfig
    model: ModelConfig
    ppo: PPOConfig
    diversity: NMDConfig | SNDConfig | None
    optimizer: OptimizerConfig
    evaluation: EvalConfig
    logging: LoggingConfig

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        if data["diversity"] is None:
            del data["diversity"]
        for (section, key), methods in _METHOD_ONLY_FIELDS.items():
            if self.method not in methods:
                del data[section][key]
        return data


_SECTIONS = {
    "env": EnvConfig,
    "model": ModelConfig,
    "ppo": PPOConfig,
    "optimizer": OptimizerConfig,
    "evaluation": EvalConfig,
    "logging": LoggingConfig,
}
_TOP_LEVEL = {"method", "seed", "jax_threefry_partitionable"}
_TUPLE_FIELDS = {"policy_hidden_dims", "speed_range", "lidar_range_range", "eval_max_speeds",
                 "max_speed_range",
                 "force_multiplier_range", "target_nmd_list"}


def _build(section: str, cls, data: Mapping[str, Any], method: str):
    fields = {f.name for f in dataclasses.fields(cls)}
    absent = {key for (sec, key), methods in _METHOD_ONLY_FIELDS.items()
              if sec == section and method not in methods}
    given = set(data)
    if missing := sorted(fields - absent - given):
        raise ValueError(f"config section '{section}' is missing required keys: {missing}")
    if unknown := sorted(given - (fields - absent)):
        raise ValueError(f"config section '{section}' has unknown keys for method "
                         f"'{method}': {unknown}")
    values = {k: (tuple(v) if k in _TUPLE_FIELDS and v is not None else v) for k, v in data.items()}
    return cls(**values, **{key: None for key in absent})


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raw[section] = value
            continue
        if section not in raw:
            raise ValueError(f"override '{dotted}' names an unknown section")
        raw[section][key] = value

    top = {k: v for k, v in raw.items() if k not in _SECTIONS and k != "diversity"}
    if unknown := sorted(set(top) - _TOP_LEVEL):
        raise ValueError(f"config has unknown top-level keys: {unknown}")
    if missing := sorted(_TOP_LEVEL - set(top)):
        raise ValueError(f"config is missing required top-level keys: {missing}")
    method = top["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if missing := sorted(set(_SECTIONS) - set(raw)):
        raise ValueError(f"config is missing sections: {missing}")

    diversity_cls = DIVERSITY_SECTIONS[method]
    if diversity_cls is None and "diversity" in raw:
        raise ValueError(f"method '{method}' has no diversity measure; remove the "
                         "'diversity' section")
    if diversity_cls is not None and "diversity" not in raw:
        raise ValueError("config is missing sections: ['diversity']")

    if not isinstance(top["jax_threefry_partitionable"], bool):
        raise ValueError("jax_threefry_partitionable must be true or false")

    return Config(
        method=method,
        seed=int(top["seed"]),
        jax_threefry_partitionable=top["jax_threefry_partitionable"],
        diversity=(None if diversity_cls is None
                   else _build("diversity", diversity_cls, raw["diversity"], method)),
        **{name: _build(name, cls, raw[name], method) for name, cls in _SECTIONS.items()},
    )


def dump_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

