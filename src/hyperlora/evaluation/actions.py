from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..algorithms.ps import zero_adapters
from ..envs.tokens import capability_context, pressure_plate_env_context


def ours(models, config, adapter, num_agents, obs_dim, hn_apply, events_factory=None):
    @jax.jit
    def act(policy_params, obs_batch, adapters):
        mean, _log_std = models.policy.apply({"params": policy_params}, obs_batch, adapters)
        return jnp.clip(mean, -1.0, 1.0)

    num_envs = config.env.num_envs

    def tokens_fn(env, obs, scaling):
        tokens = adapter.build_tokens(config, env, obs, obs_dim, num_envs, num_agents,
                                      num_agents, config.diversity.target_nmd)
        tokens = type(tokens)(**{name: getattr(tokens, name) for name in tokens.FIELDS},
                              mask=None)
        events = events_factory(env, obs) if events_factory is not None else None
        return {"tokens": tokens, "adapters": hn_apply(tokens, scaling), "events": events,
                "scaling": scaling, "step": 0,
                "prev_door_open": torch.zeros(num_envs, dtype=torch.bool),
                "plate_count": torch.zeros(num_envs, dtype=torch.long)}

    def actions_fn(obs_batch, state):
        return act(models.policy_state.params, obs_batch, state["adapters"]), state

    def after_step(env, state):
        tokens = state["tokens"]
        if not (config.model.adaptive_hypernetwork and config.env.scenario == "pressure_plate"
                and tokens.env_context is not None):
            return state

        scenario = env.scenario
        current = scenario.door_open.clone()
        newly_opened = current & ~state["prev_door_open"]

        left = torch.zeros(num_envs, dtype=torch.bool)
        right = torch.zeros(num_envs, dtype=torch.bool)
        for robot in [a for a in env.agents if "ground_robot" in a.name]:
            reach = scenario.plate_radius + robot.shape.radius + 0.05
            left |= torch.linalg.norm(robot.state.pos - scenario.plate_left.state.pos,
                                      dim=-1) < reach
            right |= torch.linalg.norm(robot.state.pos - scenario.plate_right.state.pos,
                                       dim=-1) < reach
        active = left.long() + right.long()
        second_plate = (active == 2) & (state["plate_count"] == 1)
        state["plate_count"] = torch.maximum(state["plate_count"], active)

        should = newly_opened | second_plate
        if bool(should.any()):
            if config.model.env_context.get("door_state", True):
                rebuilt = pressure_plate_env_context(config, env, num_envs, num_agents)
                tokens = type(tokens)(**{**{n: getattr(tokens, n) for n in tokens.FIELDS},
                                         "env_context": jnp.asarray(rebuilt.cpu().numpy())})
                state["tokens"] = tokens
            index = np.where(np.repeat(should.cpu().numpy(), num_agents))[0]
            batch = num_envs * num_agents

            def subset(x):
                return None if x is None else x.reshape(batch, -1)[index][:, None, :]

            new = models.hypernetwork.apply(
                {"params": models.hn_state.params},
                subset(tokens.task), subset(tokens.context), None, None, None, None,
                subset(tokens.env_context), None, state["scaling"])
            state["adapters"] = {k: v.at[index].set(new[k]) for k, v in state["adapters"].items()}
            state["requeries"] = state.get("requeries", 0) + int(len(index))
        state["prev_door_open"] = current
        return state

    return actions_fn, tokens_fn, after_step


def ps(models, config, adapter, num_agents, obs_dim, input_dim, action_dim):
    @jax.jit
    def act(policy_params, obs_batch, adapters):
        mean, _log_std = models.policy.apply({"params": policy_params}, obs_batch, adapters)
        return jnp.clip(mean, -1.0, 1.0)

    num_envs = config.env.num_envs
    batch = num_envs * num_agents
    adapters = zero_adapters(batch, input_dim, tuple(config.model.policy_hidden_dims), action_dim)

    def tokens_fn(env, obs, scaling):
        suffix = None
        if adapter.env_context is not None and input_dim > obs_dim:
            context = adapter.env_context(config, env, num_envs, num_agents)
            suffix = jnp.asarray(context.cpu().numpy()).reshape(batch, -1)
        return {"suffix": suffix}

    def actions_fn(obs_batch, state):
        if state["suffix"] is not None:
            obs_batch = jnp.concatenate([obs_batch, state["suffix"]], axis=-1)
        return act(models.policy_state.params, obs_batch, adapters), state

    return actions_fn, tokens_fn


def dico(models, config, num_agents):
    @jax.jit
    def act(policy_params, obs_batch, agent_ids):
        mean, _log_std = models.policy.apply({"params": policy_params}, obs_batch, agent_ids,
                                             diversity_scaling=1.0)
        return jnp.clip(mean, -1.0, 1.0)

    num_envs = config.env.num_envs
    agent_ids = jnp.asarray(np.tile(np.arange(num_agents), num_envs) % models.policy.num_agents)

    def tokens_fn(env, obs, scaling):
        suffix = None
        if config.env.scenario == "dispersion" and config.model.use_capability_context:
            speeds = np.array([float(getattr(env.agents[i], "_max_speed"))
                               for i in range(num_agents)], dtype=np.float32)
            suffix = jnp.asarray(np.tile(speeds, num_envs)[:, np.newaxis])
        return {"suffix": suffix}

    def actions_fn(obs_batch, state):
        if state["suffix"] is not None:
            obs_batch = jnp.concatenate([obs_batch, state["suffix"]], axis=-1)
        return act(models.policy_state.params, obs_batch, agent_ids), state

    return actions_fn, tokens_fn


def cash(models, config, adapter, num_agents, obs_dim, input_dim):
    @jax.jit
    def act(policy_params, obs_batch, capabilities, hidden):
        new_hidden, (mean_seq, _log_std) = models.policy.apply(
            {"params": policy_params}, hidden,
            (obs_batch[None, ...], jnp.zeros((1, obs_batch.shape[0]), dtype=bool),
             capabilities[None, ...]))
        return new_hidden, jnp.clip(mean_seq[0], -1.0, 1.0)

    num_envs = config.env.num_envs
    batch = num_envs * num_agents
    def tokens_fn(env, obs, scaling):
        from ..training.trainer import token_dims

        width = token_dims(config, obs_dim)["context"]
        if config.env.scenario == "dispersion" and width <= 1:
            capabilities = jnp.zeros((batch, width))
        else:
            context = capability_context(config, env, num_envs, num_agents)
            capabilities = jnp.asarray(context.cpu().numpy()).reshape(batch, -1)
        suffix = None
        if adapter.env_context is not None and input_dim > obs_dim:
            env_context = adapter.env_context(config, env, num_envs, num_agents)
            suffix = jnp.asarray(env_context.cpu().numpy()).reshape(batch, -1)
        return {"capabilities": capabilities, "suffix": suffix,
                "hidden": jnp.zeros((batch, config.model.gru_hidden_dim))}

    def actions_fn(obs_batch, state):
        if state["suffix"] is not None:
            obs_batch = jnp.concatenate([obs_batch, state["suffix"]], axis=-1)
        hidden, actions = act(models.policy_state.params, obs_batch, state["capabilities"],
                              state["hidden"])
        return actions, {**state, "hidden": hidden}

    return actions_fn, tokens_fn

