"""
Custom evaluation utilities for HyperMARL baseline tracking completion rate, food found, and average reward.
"""

import jax
import jax.numpy as jnp
import numpy as np
import distrax
from typing import Dict, Any, Tuple


def evaluate_policy_with_metrics(
    env,
    network,
    params,
    num_eval_episodes: int,
    num_envs: int,
    max_eval_steps: int,
    seed: int = 42,
    deterministic: bool = True,
    scenario_name: str = "dispersion_vmas",
) -> Dict[str, float]:
    """
    Evaluate a policy and track custom metrics: completion rate, food found, average reward.

    Args:
        env: Vectorized VMAS environment
        network: Policy network
        params: Network parameters
        num_eval_episodes: Number of evaluation episodes per environment
        num_envs: Number of parallel environments
        max_eval_steps: Maximum steps per episode
        seed: Random seed for evaluation
        deterministic: Whether to use deterministic actions
        scenario_name: Name of the scenario ("dispersion_vmas", "dispersion", "reverse_transport")

    Returns:
        Dictionary containing:
            - completion_rate: % of episodes that completed successfully
            - food_found: Average number of food items collected (dispersion only)
            - avg_reward: Average episode reward
            - avg_episode_length: Average episode length
            - std_reward: Standard deviation of rewards
    """
    # Initialize metrics storage
    total_episodes = 0
    completed_episodes = 0
    total_food_collected = 0
    episode_rewards = []
    episode_lengths = []

    # Determine if this is a dispersion scenario
    is_dispersion = "dispersion" in scenario_name.lower()
    num_food_per_env = env.num_agents if is_dispersion else 0

    # Get number of agents and envs from environment
    num_agents = env.num_agents
    actual_num_envs = env.num_envs  # Use actual env's num_envs, not parameter

    # Get action dimension
    action_dim = env.action_dim

    # Calculate observation and critic sizes
    # Note: observations do NOT include agent IDs (agent IDs computed from batch position)
    obs_size = env.observation_size
    critic_obs_size = obs_size * num_agents

    # JIT-compiled action selection function
    @jax.jit
    def select_action(params_inner, obs, obs_critic, rng_key):
        """Select action using the policy network."""
        means, log_stds, _ = network.apply(params_inner, obs, obs_critic)
        if deterministic:
            # Use means for deterministic evaluation
            action = means
        else:
            # Sample from the distribution
            stds = jnp.exp(log_stds)
            pi = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)
            action = pi.sample(seed=rng_key)
        # Clip actions to valid range
        action = jnp.clip(action, -1.0, 1.0)
        return action

    # Helper function to extract agent observations
    def get_agent_obs(obs_dict):
        """Extract agent observations from dict format."""
        # obs_dict has agents as keys, values have shape (num_envs, obs_size)
        # Stack observations for all agents
        obs_array = np.array([obs_dict[agent] for agent in sorted(obs_dict.keys())])
        # obs_array shape after stacking: (num_agents, num_envs, obs_size)
        # Swap axes to get: (num_envs, num_agents, obs_size)
        obs_array = np.swapaxes(obs_array, 0, 1)
        return obs_array

    # Helper function to create global observation
    def create_global_obs(obs_array):
        """Create global observation by concatenating all agent observations."""
        # obs_array shape: (num_envs, num_agents, obs_size)
        # Flatten agent dimension: (num_envs, num_agents * obs_size)
        return obs_array.reshape(obs_array.shape[0], -1)

    # Run evaluation episodes
    rng = jax.random.PRNGKey(seed)

    for episode_idx in range(num_eval_episodes):
        # Reset environment
        reset_seed = seed + episode_idx * 1000
        obs_dict, _ = env.reset(seed=reset_seed)

        # Initialize episode tracking
        episode_reward = np.zeros(actual_num_envs)
        episode_length = np.zeros(actual_num_envs, dtype=np.int32)
        episode_done = np.zeros(actual_num_envs, dtype=bool)
        food_collected_per_env = np.zeros(actual_num_envs, dtype=np.int32)

        for step in range(max_eval_steps):
            # Extract observations (no agent IDs in observations)
            obs_array = get_agent_obs(obs_dict)

            # Create global observation for critic
            global_obs = create_global_obs(obs_array)

            # Flatten observations for all agents across all envs
            # Shape: (num_envs * num_agents, obs_size)
            obs_flat = obs_array.reshape(-1, obs_size)

            # Repeat global obs for each agent
            # Shape: (num_envs * num_agents, critic_obs_size)
            global_obs_flat = np.repeat(global_obs, num_agents, axis=0)

            # Convert to JAX arrays
            obs_jax = jnp.array(obs_flat)
            global_obs_jax = jnp.array(global_obs_flat)

            # Select actions
            rng, action_rng = jax.random.split(rng)
            actions_flat = select_action(params, obs_jax, global_obs_jax, action_rng)
            actions_np = np.array(actions_flat)

            # Reshape actions based on action dimension
            if action_dim > 1:
                # Continuous actions: (num_envs * num_agents, action_dim) -> (num_envs, num_agents, action_dim)
                actions = actions_np.reshape(actual_num_envs, num_agents, action_dim)
            else:
                # Discrete actions: (num_envs * num_agents,) -> (num_envs, num_agents)
                actions = actions_np.reshape(actual_num_envs, num_agents)

            # Step environment
            # Convert actions to environment format
            if action_dim > 1:
                # For continuous actions
                action_dict = {
                    agent: actions[:, i, :]
                    for i, agent in enumerate(sorted(obs_dict.keys()))
                }
            else:
                # For discrete actions
                action_dict = {
                    agent: actions[:, i]
                    for i, agent in enumerate(sorted(obs_dict.keys()))
                }
            obs_dict, rewards, dones, truncs, infos = env.step(action_dict)

            # Aggregate rewards across agents (sum per environment)
            # rewards is a dict: {agent_name: array of shape (num_envs,)}
            reward_array = np.array(
                [rewards[agent] for agent in sorted(rewards.keys())]
            ).T
            # reward_array shape: (num_envs, num_agents)
            reward_per_env = reward_array.mean(axis=1)  # Average across agents

            # Check if episodes are done
            # dones is a dict: {agent_name: array of shape (num_envs,)}
            done_array = np.array([dones[agent] for agent in sorted(dones.keys())]).T
            # done_array shape: (num_envs, num_agents)
            done_per_env = done_array.all(axis=1)  # All agents must be done

            # Update episode metrics for non-completed episodes
            for env_idx in range(actual_num_envs):
                if not episode_done[env_idx]:
                    episode_reward[env_idx] += reward_per_env[env_idx]
                    episode_length[env_idx] += 1

                    # Check if episode just completed
                    if done_per_env[env_idx]:
                        episode_done[env_idx] = True

            # Track food collected for dispersion scenarios
            if is_dispersion and step == max_eval_steps - 1:
                # For dispersion, check how many food items were eaten
                # Access the underlying VMAS environment
                if hasattr(env, "_env"):
                    vmas_env = env._env
                elif hasattr(env, "env"):
                    vmas_env = env.env
                else:
                    vmas_env = env

                if hasattr(vmas_env, "world"):
                    for landmark in vmas_env.world.landmarks:
                        if hasattr(landmark, "eaten"):
                            # landmark.eaten is a tensor of shape (num_envs,)
                            food_collected_per_env += (
                                landmark.eaten.cpu().numpy().astype(np.int32)
                            )

            # Break if all environments are done
            if episode_done.all():
                break

        # Record metrics for this batch of episodes
        for env_idx in range(actual_num_envs):
            total_episodes += 1
            if episode_done[env_idx]:
                completed_episodes += 1
            episode_rewards.append(episode_reward[env_idx])
            episode_lengths.append(episode_length[env_idx])
            if is_dispersion:
                total_food_collected += food_collected_per_env[env_idx]

    # Calculate final metrics
    completion_rate = (completed_episodes / total_episodes) * 100.0
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_episode_length = np.mean(episode_lengths)

    metrics = {
        "completion_rate": completion_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_episode_length": avg_episode_length,
        "total_episodes": total_episodes,
        "completed_episodes": completed_episodes,
    }

    # Add food-specific metrics for dispersion scenarios
    if is_dispersion:
        total_food_available = total_episodes * num_food_per_env
        food_found_percentage = (
            (total_food_collected / total_food_available) * 100.0
            if total_food_available > 0
            else 0.0
        )
        metrics["food_found_total"] = total_food_collected
        metrics["food_found_percentage"] = food_found_percentage
        metrics["avg_food_per_episode"] = total_food_collected / total_episodes

    return metrics
