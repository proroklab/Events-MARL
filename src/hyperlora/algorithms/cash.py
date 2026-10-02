from __future__ import annotations

from functools import partial

import distrax
import jax
import jax.numpy as jnp


def make_train_step(shared_policy, critic):
    @partial(jax.jit, static_argnums=(15, 16))
    def train_step(
        policy_state,
        critic_state,
        obs_sequence,
        capability_sequence,
        global_state_sequence,
        actions_sequence,
        old_log_probs_sequence,
        advantages_sequence,
        returns_sequence,
        init_hidden_state,
        init_critic_hidden,
        dones_sequence,
        clip_epsilon,
        entropy_coef,
        value_loss_coef,
        num_agents,
        current_num_agents,
    ):
        def loss_fn(policy_params, critic_params):
            num_steps, num_actors = obs_sequence.shape[0], obs_sequence.shape[1]
            num_envs = num_actors // num_agents

            dones_shifted = jnp.concatenate(
                [jnp.zeros((1, dones_sequence.shape[1]), dtype=bool), dones_sequence[:-1]],
                axis=0)
            dones_for_actor = jnp.broadcast_to(
                dones_shifted[:, :, None], (num_steps, num_envs, num_agents)
            ).reshape(num_steps, num_actors)

            policy_x = (obs_sequence, dones_for_actor, capability_sequence)
            _, (mean_all, log_std_all) = shared_policy.apply(
                {"params": policy_params}, init_hidden_state, policy_x)
            std_all = jnp.exp(log_std_all)

            base_dist = distrax.Normal(loc=mean_all.reshape(-1, mean_all.shape[-1]),
                                       scale=std_all.reshape(-1, std_all.shape[-1]))
            dist = distrax.Transformed(base_dist, distrax.Tanh())
            actions_flat = actions_sequence.reshape(-1, actions_sequence.shape[-1])
            new_log_probs = dist.log_prob(actions_flat).sum(axis=-1).reshape(
                num_steps, num_actors)
            entropy_all = base_dist.entropy().sum(axis=-1).reshape(num_steps, num_actors)

            new_log_probs = jnp.nan_to_num(new_log_probs, nan=-100.0, posinf=-100.0,
                                           neginf=-100.0)
            new_log_probs = jnp.clip(new_log_probs, -100.0, 100.0)

            log_prob_diff = jnp.clip(new_log_probs - old_log_probs_sequence, -3.0, 3.0)
            ratio = jnp.exp(log_prob_diff)
            surr1 = ratio * advantages_sequence
            surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_sequence
            policy_loss_all = -jnp.minimum(surr1, surr2)

            _, values_all = critic.apply({"params": critic_params}, init_critic_hidden,
                                         (global_state_sequence, dones_sequence))
            value_loss_all = jnp.square(values_all.reshape(values_all.shape[0], -1)
                                        - returns_sequence)

            agent_mask_template = jnp.arange(num_agents, dtype=jnp.int32) < current_num_agents
            valid_agent_mask = jnp.broadcast_to(
                agent_mask_template[None, None, :], (num_steps, num_envs, num_agents)
            ).reshape(num_steps, num_actors)
            denom = jnp.maximum(valid_agent_mask.sum(), 1.0)

            policy_loss = (policy_loss_all * valid_agent_mask).sum() / denom
            value_loss = (value_loss_all * valid_agent_mask).sum() / denom
            entropy = (entropy_all * valid_agent_mask).sum() / denom

            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jnp.mean((new_log_probs - old_log_probs_sequence) ** 2),
                "mean_ratio": jnp.mean(ratio),
                "advantages_mean": jnp.mean(advantages_sequence),
                "advantages_std": jnp.std(advantages_sequence),
                "advantages_max": jnp.max(jnp.abs(advantages_sequence)),
                "mean_std": jnp.mean(std_all),
                "min_std": jnp.min(std_all),
                "max_std": jnp.max(std_all),
            }

        (loss, info), grads = jax.value_and_grad(
            lambda p, c: loss_fn(p, c), argnums=(0, 1), has_aux=True
        )(policy_state.params, critic_state.params)
        policy_grads, critic_grads = grads

        info["policy_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads)))
        info["hn_grad_norm"] = jnp.asarray(0.0)
        info["critic_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads)))

        nan_to_zero = lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g)
        policy_grads = jax.tree_util.tree_map(nan_to_zero, policy_grads)
        critic_grads = jax.tree_util.tree_map(nan_to_zero, critic_grads)

        return (policy_state.apply_gradients(grads=policy_grads),
                critic_state.apply_gradients(grads=critic_grads), info)

    return train_step


def sequence_minibatches(num_envs: int, num_minibatches: int, num_agents: int):
    if num_envs % num_minibatches:
        raise ValueError(f"{num_envs} environments do not split into {num_minibatches} minibatches")
    per_batch = num_envs // num_minibatches
    out = []
    for i in range(num_minibatches):
        envs = jnp.arange(i * per_batch, (i + 1) * per_batch)
        agents = (envs[:, None] * num_agents + jnp.arange(num_agents)[None, :]).flatten()
        out.append((envs, agents))
    return out

