from __future__ import annotations

import jax


def make_jitted(policy, hypernetwork, critic):
    @jax.jit
    def get_static_adapters(hn_params, task_batch, context_batch, lidar_batch,
                            food_positions_batch, agent_positions_batch, target_nmd_batch,
                            env_context_batch, mask, diversity_scaling=1.0):
        return hypernetwork.apply(
            {"params": hn_params}, task_batch, context_batch, lidar_batch, food_positions_batch,
            agent_positions_batch, target_nmd_batch, env_context_batch, mask, diversity_scaling,
        )

    @jax.jit
    def get_actions_and_log_probs(policy_params, obs_batch, adapters_dict, rng_key):
        action, log_prob, mean, std = policy.apply(
            {"params": policy_params}, obs_batch, adapters_dict, rng_key,
            method=policy.get_action_and_log_prob,
        )
        return action, log_prob, None, mean, std

    @jax.jit
    def get_value(critic_params, global_state):
        return critic.apply({"params": critic_params}, global_state)

    return get_static_adapters, get_actions_and_log_probs, get_value


def make_dico_jitted(policy, critic):
    @jax.jit
    def get_actions_and_log_probs(policy_params, obs_batch, agent_ids, diversity_scaling, rng_key):
        action, log_prob, mean, std = policy.apply(
            {"params": policy_params}, obs_batch, agent_ids, diversity_scaling, rng_key,
            method=policy.get_action_and_log_prob,
        )
        return action, log_prob, None, mean, std

    @jax.jit
    def get_value(critic_params, global_state):
        return critic.apply({"params": critic_params}, global_state)

    return get_actions_and_log_probs, get_value

