from __future__ import annotations

from functools import partial

import distrax
import jax
import jax.numpy as jnp


def zero_adapters(batch_size: int, obs_dim: int, hidden_dims, action_dim: int) -> dict:
    adapters = {}
    input_dim = obs_dim
    for i, output_dim in enumerate(hidden_dims):
        adapters[f"A{i + 1}"] = jnp.zeros((batch_size, 0, input_dim))
        adapters[f"B{i + 1}"] = jnp.zeros((batch_size, output_dim, 0))
        input_dim = output_dim
    final = len(hidden_dims) + 1
    adapters[f"A{final}"] = jnp.zeros((batch_size, 0, hidden_dims[-1]))
    adapters[f"B{final}"] = jnp.zeros((batch_size, action_dim, 0))
    return adapters


def make_train_step(shared_policy, critic):
    @partial(jax.jit, static_argnums=(12, 13))
    def train_step(
        policy_state,
        critic_state,
        obs_batch,
        global_state_batch,
        actions_batch,
        old_log_probs_batch,
        advantages_batch,
        returns_batch,
        adapters_dict,
        clip_epsilon,
        entropy_coef,
        value_loss_coef,
        num_agents,
        current_num_agents,
    ):
        def loss_fn(policy_params, critic_params):
            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, adapters_dict
            )
            std = jnp.exp(log_std)

            base_dist = distrax.Normal(mean, std)
            dist = distrax.Transformed(base_dist, distrax.Tanh())
            new_log_probs = dist.log_prob(actions_batch).sum(axis=-1)
            entropy_per_agent = base_dist.entropy().sum(axis=-1)

            new_log_probs = jnp.nan_to_num(new_log_probs, nan=-100.0, posinf=-100.0, neginf=-100.0)
            new_log_probs = jnp.clip(new_log_probs, -100.0, 100.0)

            log_prob_diff = jnp.clip(new_log_probs - old_log_probs_batch, -3.0, 3.0)
            ratio = jnp.exp(log_prob_diff)
            surr1 = ratio * advantages_batch
            surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch
            policy_loss_per_agent = -jnp.minimum(surr1, surr2)

            batch_size_total = obs_batch.shape[0]
            agent_mask_template = jnp.arange(num_agents) < current_num_agents
            valid_agent_mask = jnp.tile(agent_mask_template, batch_size_total // num_agents)

            policy_loss_masked = policy_loss_per_agent * valid_agent_mask
            policy_loss = policy_loss_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

            entropy_masked = entropy_per_agent * valid_agent_mask
            entropy = entropy_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

            policy_loss = policy_loss + 0.001 * jnp.mean(jnp.square(mean))

            global_state_envs = global_state_batch[::num_agents]
            values = critic.apply({"params": critic_params}, global_state_envs)
            values_flat = values.reshape(-1)

            value_loss_per_agent = (values_flat - returns_batch) ** 2
            value_loss_masked = value_loss_per_agent * valid_agent_mask
            value_loss = value_loss_masked.sum() / jnp.maximum(valid_agent_mask.sum(), 1.0)

            total_loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

            info_dict = {
                "total_loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jnp.mean((new_log_probs - old_log_probs_batch) ** 2),
                "mean_ratio": jnp.mean(ratio),
                "advantages_mean": jnp.mean(advantages_batch),
                "advantages_std": jnp.std(advantages_batch),
                "advantages_max": jnp.max(jnp.abs(advantages_batch)),
                "adapter_norm": 0.0,
                "mean_std": jnp.mean(std),
                "min_std": jnp.min(std),
                "max_std": jnp.max(std),
            }
            return total_loss, info_dict

        (loss, info), grads = jax.value_and_grad(
            lambda p, c: loss_fn(p, c), argnums=(0, 1), has_aux=True
        )(policy_state.params, critic_state.params)
        policy_grads, critic_grads = grads

        info["policy_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads)))
        info["hn_grad_norm"] = 0.0
        info["critic_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads)))

        nan_to_zero = lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g)
        policy_grads = jax.tree_util.tree_map(nan_to_zero, policy_grads)
        critic_grads = jax.tree_util.tree_map(nan_to_zero, critic_grads)

        return (
            policy_state.apply_gradients(grads=policy_grads),
            critic_state.apply_gradients(grads=critic_grads),
            info,
        )

    return train_step

