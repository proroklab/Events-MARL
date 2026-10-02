from __future__ import annotations

from functools import partial

import distrax
import jax
import jax.numpy as jnp


def make_train_step(shared_policy, critic):
    def train_step(policy_state, critic_state, obs_batch, global_state_batch, actions_batch,
                   old_log_probs_batch, advantages_batch, returns_batch, agent_ids_batch,
                   diversity_scaling, clip_epsilon, entropy_coef, value_loss_coef, num_agents,
                   current_num_agents):
        def loss_fn(policy_params, critic_params):
            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, agent_ids_batch, diversity_scaling)
            std = jnp.exp(log_std)

            base_dist = distrax.Normal(mean, std)
            dist = distrax.Transformed(base_dist, distrax.Tanh())
            actions_clipped = jnp.clip(actions_batch, -0.9999, 0.9999)
            new_log_probs = dist.log_prob(actions_clipped).sum(axis=-1)
            new_log_probs = jnp.nan_to_num(new_log_probs, nan=-10.0, posinf=-10.0, neginf=-10.0)
            entropy_per_agent = base_dist.entropy().sum(axis=-1)

            log_prob_diff = jnp.clip(new_log_probs - old_log_probs_batch, -3.0, 3.0)
            ratio = jnp.nan_to_num(jnp.exp(log_prob_diff), nan=1.0, posinf=1.0, neginf=1.0)

            surr1 = ratio * advantages_batch
            surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch
            policy_loss_per_agent = jnp.nan_to_num(
                -jnp.minimum(surr1, surr2), nan=0.0, posinf=0.0, neginf=0.0)

            batch_size_total = obs_batch.shape[0]
            agent_mask_template = jnp.arange(num_agents) < current_num_agents
            valid_agent_mask = jnp.tile(agent_mask_template, batch_size_total // num_agents)

            policy_loss = ((policy_loss_per_agent * valid_agent_mask).sum()
                           / jnp.maximum(valid_agent_mask.sum(), 1.0))
            entropy = ((entropy_per_agent * valid_agent_mask).sum()
                       / jnp.maximum(valid_agent_mask.sum(), 1.0))

            values = critic.apply({"params": critic_params}, global_state_batch)
            values = jnp.clip(values, -100.0, 100.0)
            values = jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
            values_flat = values.reshape(-1)

            returns_clipped = jnp.clip(returns_batch, -100.0, 100.0)
            returns_clipped = jnp.nan_to_num(returns_clipped, nan=0.0, posinf=100.0,
                                             neginf=-100.0)
            value_loss = (((values_flat - returns_clipped) ** 2 * valid_agent_mask).sum()
                          / jnp.maximum(valid_agent_mask.sum(), 1.0))

            total_loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss
            return total_loss, {
                "total_loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": jnp.mean((new_log_probs - old_log_probs_batch) ** 2),
                "mean_std": jnp.mean(std),
                "min_std": jnp.min(std),
                "max_std": jnp.max(std),
                "mean_ratio": jnp.mean(ratio),
                "advantages_mean": jnp.mean(advantages_batch),
                "advantages_std": jnp.std(advantages_batch),
                "advantages_max": jnp.max(jnp.abs(advantages_batch)),
                "adapter_norm": jnp.asarray(0.0),
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
        false = jnp.array(False)
        info["policy_grad_has_nan"] = jax.tree_util.tree_reduce(
            lambda acc, g: acc | jnp.isnan(g).any(), policy_grads, false)
        info["critic_grad_has_nan"] = jax.tree_util.tree_reduce(
            lambda acc, g: acc | jnp.isnan(g).any(), critic_grads, false)

        nan_to_zero = lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g)
        policy_grads = jax.tree_util.tree_map(nan_to_zero, policy_grads)
        critic_grads = jax.tree_util.tree_map(nan_to_zero, critic_grads)

        return (policy_state.apply_gradients(grads=policy_grads),
                critic_state.apply_gradients(grads=critic_grads), info)

    return train_step


def make_ppo_update(shared_policy, critic):
    train_step = make_train_step(shared_policy, critic)

    @partial(jax.jit, static_argnames=("num_minibatches", "ppo_epochs", "num_agents",
                                       "current_num_agents"))
    def ppo_update(policy_state, critic_state, obs_batch, global_state_batch, actions_batch,
                   old_log_probs_batch, advantages_batch, returns_batch, agent_ids_batch,
                   diversity_scaling, clip_epsilon, entropy_coef, value_loss_coef, *,
                   num_minibatches, ppo_epochs, num_agents, current_num_agents):
        mb_size = obs_batch.shape[0] // num_minibatches
        gs_mb_size = global_state_batch.shape[0] // num_minibatches
        split = lambda x, chunk: x.reshape((num_minibatches, chunk) + x.shape[1:])

        batches = (split(obs_batch, mb_size), split(global_state_batch, gs_mb_size),
                   split(actions_batch, mb_size), split(old_log_probs_batch, mb_size),
                   split(advantages_batch, mb_size), split(returns_batch, mb_size),
                   split(agent_ids_batch, mb_size))

        def mb_step(carry, batch):
            ps, cs = carry
            obs_i, gs_i, acts_i, old_lp_i, adv_i, ret_i, aids_i = batch
            ps, cs, info = train_step(ps, cs, obs_i, gs_i, acts_i, old_lp_i, adv_i, ret_i, aids_i,
                                      diversity_scaling, clip_epsilon, entropy_coef,
                                      value_loss_coef, num_agents, current_num_agents)
            return (ps, cs), info

        def epoch_step(carry, _):
            carry, infos = jax.lax.scan(mb_step, carry, batches)
            return carry, jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), infos)

        (policy_state, critic_state), epoch_infos = jax.lax.scan(
            epoch_step, (policy_state, critic_state), xs=None, length=ppo_epochs)
        return policy_state, critic_state, jax.tree_util.tree_map(lambda x: x[-1], epoch_infos)

    return ppo_update

