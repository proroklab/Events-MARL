"""
MLP Hypernetworks for MAPPO with shared weights.


HyperMARL generates both actor and critic parameters using a linear hypernetwork. Identical to linear HyperMARL but with MLP.
"""

from enum import Enum
import multiprocessing as mp
import time
import yaml
import argparse
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Tuple
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.struct import dataclass
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

import wandb

# Import local modules
from mappo import get_update_fn
from vmas_wrapper_utils import make_env

# Try to import custom evaluation utilities
try:
    from hypermarl_eval_utils import evaluate_policy_with_metrics

    CUSTOM_EVAL_AVAILABLE = True
except ImportError:
    CUSTOM_EVAL_AVAILABLE = False
    print("Warning: Custom evaluation utilities not found.")

# These are not used in the core training loop, set to stubs
BASELINES_AVAILABLE = False
run_eval_jax = None


def calculate_team_diversity(*args, **kwargs):
    pass


def extract_and_save_agent_embeddings(*args, **kwargs):
    pass


def log_eval_metrics(*args, **kwargs):
    pass


def log_train_metrics(*args, **kwargs):
    pass


def log_videos(*args, **kwargs):
    pass


# Set up multiprocessing with fork server for better performance
forkserver_available = "forkserver" in mp.get_all_start_methods()
start_method = "forkserver" if forkserver_available else "spawn"
mp.set_start_method(start_method, force=True)


@dataclass
class EpisodeStatistics:
    """
    Tracks statistics about episodes during training.

    Attributes:
        episode_returns: Cumulative rewards for ongoing episodes
        episode_lengths: Number of steps taken in ongoing episodes
        returned_episode_returns: Returns of completed episodes
        returned_episode_lengths: Lengths of completed episodes
    """

    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class HyperNetType(Enum):
    """
    Enum to distinguish between actor and critic hypernetworks.

    Different types may use different initialization strategies.
    """

    ACTOR = 0
    CRITIC = 1


def check_rows_orthogonal(A):
    """
    Check if the rows of a matrix are approximately orthogonal.

    Args:
        A: Input matrix to check

    Returns:
        Boolean indicating whether rows are orthogonal
    """
    dot_products = jnp.dot(A, A.T)
    off_diagonal_elements = dot_products - jnp.diag(jnp.diag(dot_products))
    is_orthogonal = jnp.all(jnp.isclose(off_diagonal_elements, 0, atol=0.0001))
    return is_orthogonal


class MLPHyperNetwork(nn.Module):
    """
    MLP HyperNetwork for generating weights and biases of all layers in the network.

    Uses multi-layer perceptrons to transform agent embeddings into network parameters,
    allowing for more complex weight generation than linear hypernetworks.

    Attributes:
        output_dims: List of (input_dim, output_dim) tuples for each layer
        hypernet_type: Type of hypernetwork (actor or critic)
        init_scale: Initialization scale factor
        use_bias: Whether to use bias in the hypernetwork
        hidden_dims: Sizes of hidden layers in the MLP hypernetwork
    """

    output_dims: List[Tuple[int, int]]
    hypernet_type: HyperNetType
    init_scale: float = np.sqrt(2)  # default when using relu
    use_bias: bool = True
    hidden_dims: List[int] = (64,)  # Default hidden layer sizes for the MLP

    @staticmethod
    def hypernet_init(gain, fan_in, fan_out):
        """
        Custom initialization function for the hypernetwork weights.
        Simply takes an initialization function and applies it to each agent's generated weights e.g. orthogonal init weights are commonly used, so this ensures that this generates orthogonal weights for each agent at init.

        Args:
            gain: Scaling factor for weights
            fan_in: Input dimension
            fan_out: Output dimension

        Returns:
            Initialization function
        """

        def weight_init(key, shape, dtype):
            init = jax.nn.initializers.orthogonal(gain)
            batched_init = jax.vmap(init, in_axes=(0, None, None))

            # Create a batch of keys (one per agent)
            batch_size = shape[0]
            keys = jax.random.split(key, num=batch_size)

            weights = batched_init(keys, (fan_in, fan_out), dtype)
            return weights.reshape(shape)

        return weight_init

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the hypernetwork.

        Args:
            x: Agent embeddings

        Returns:
            weight_heads: Generated weights for each layer
            bias_heads: Generated biases for each layer
        """
        # Layer-specific hypernets for weights and biases
        weight_heads = []
        bias_heads = []

        for i, (input_dim, output_dim) in enumerate(self.output_dims):
            weight_dim = input_dim * output_dim
            bias_dim = output_dim

            is_final_layer = i == len(self.output_dims) - 1

            # Final layers - different gain values for critic and actor
            if is_final_layer and self.hypernet_type == HyperNetType.ACTOR:
                gain = 0.01  # Small gain for actor output layer
            elif is_final_layer and self.hypernet_type == HyperNetType.CRITIC:
                gain = 1.0  # Standard gain for critic output layer
            else:
                # Usually use np.sqrt(2) for ReLU in the non-final layers
                gain = self.init_scale

            # MLP for generating weights for this layer
            weight_mlp = x
            for hidden_dim in self.hidden_dims:
                weight_mlp = nn.Dense(hidden_dim, use_bias=self.use_bias)(weight_mlp)
                weight_mlp = nn.relu(weight_mlp)
            weight_head = nn.Dense(
                weight_dim,
                use_bias=self.use_bias,
                kernel_init=self.hypernet_init(gain, input_dim, output_dim),
                bias_init=nn.initializers.zeros,
            )(weight_mlp)

            # MLP for generating biases for this layer
            bias_mlp = x
            for hidden_dim in self.hidden_dims:
                bias_mlp = nn.Dense(hidden_dim, use_bias=self.use_bias)(bias_mlp)
                bias_mlp = nn.relu(bias_mlp)
            bias_head = nn.Dense(
                bias_dim,
                use_bias=self.use_bias,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(bias_mlp)

            weight_heads.append(weight_head)
            bias_heads.append(bias_head)

        return weight_heads, bias_heads


class ActorCritic(nn.Module):
    """
    Actor-Critic network with centralized critic using MLP hypernetworks.

    Hypernetworks generate the weights for both actor and critic networks based on
    agent embeddings, allowing parameter sharing with agent-specific adaptations.

    Attributes:
        action_dim: Dimension of action space
        num_agents: Number of agents in the environment
        actor_layers: Sizes of hidden layers for actor network
        critic_layers: Sizes of hidden layers for critic network
        embedding_dim: Dimension of agent embeddings
        observation_dim: Dimension of agent observations
        critic_obs_size: Dimension of critic input (global observation)
        init_scale: Initialization scale factor
        activation: Activation function to use (tanh or relu)
        use_agent_id_embeddings: Whether to learn agent embeddings
        use_bias_in_hypernet: Whether to use bias in hypernetworks
        hypernet_hidden_dims: Sizes of hidden layers in hypernetworks
    """

    action_dim: int
    num_agents: int
    actor_layers: List[int]
    critic_layers: List[int]
    embedding_dim: int
    observation_dim: int
    critic_obs_size: int
    init_scale: float
    activation: str = "tanh"
    use_agent_id_embeddings: bool = False
    use_bias_in_hypernet: bool = False
    action_space_type: str = "continuous"
    hypernet_hidden_dims: List[int] = (
        64,
    )  # Hidden layer sizes for the MLP hypernetwork

    def setup(self):
        """Initialize model components."""
        # Set activation function based on configuration
        self.activation_fn = jax.nn.relu if self.activation == "relu" else jax.nn.tanh

        # Initialize agent embeddings (learnable or one-hot)
        self.agent_embeddings = (
            self.param(
                "agent_embeddings",
                nn.initializers.orthogonal(self.init_scale),
                (self.num_agents, self.embedding_dim),
            )
            if self.use_agent_id_embeddings
            else jnp.eye(self.num_agents)
        )

        # Compute dimensions for all layers in actor and critic networks
        # For continuous actions, output mean and log_std for each action dimension
        actor_output_dim = (
            self.action_dim * 2
            if self.action_space_type == "continuous"
            else self.action_dim
        )
        self.actor_output_dims = self._compute_output_dims(
            self.observation_dim, self.actor_layers, actor_output_dim
        )
        self.critic_output_dims = self._compute_output_dims(
            self.critic_obs_size, self.critic_layers, 1
        )

        # Create hypernetworks for actor and critic
        self.actor_hypernet = MLPHyperNetwork(
            output_dims=self.actor_output_dims,
            init_scale=self.init_scale,
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.ACTOR,
            hidden_dims=self.hypernet_hidden_dims,
        )
        self.critic_hypernet = MLPHyperNetwork(
            output_dims=self.critic_output_dims,
            init_scale=self.init_scale,
            use_bias=self.use_bias_in_hypernet,
            hypernet_type=HyperNetType.CRITIC,
            hidden_dims=self.hypernet_hidden_dims,
        )

    def _compute_output_dims(self, input_dim, layers, final_dim):
        """
        Compute dimensions for each layer in the network.

        Args:
            input_dim: Input dimension
            layers: List of hidden layer sizes
            final_dim: Output dimension

        Returns:
            List of (input_dim, output_dim) tuples for each layer
        """
        output_dims = []
        current_dim = input_dim

        for layer_size in layers + (final_dim,):
            output_dims.append((current_dim, layer_size))
            current_dim = layer_size

        return output_dims

    @nn.compact
    def __call__(self, x, x_critic):
        """
        Forward pass through the actor-critic network.

        Args:
            x: Agent's observation (WITHOUT agent ID)
            x_critic: Global observation for critic

        Returns:
            means: Action means for continuous actions
            log_stds: Log standard deviations for continuous actions
            critic_outputs: Value function estimate
        """
        # Compute agent ID from batch position
        # Observations are ordered: [env0_agent0, env0_agent1, ..., env0_agentN, env1_agent0, ...]
        batch_size = x.shape[0]
        agent_id = jnp.arange(batch_size) % self.num_agents

        # Apply networks using hypernetwork-generated weights
        means, log_stds, critic_outputs = self._apply_networks(x, x_critic, agent_id)
        return means, log_stds, jnp.squeeze(critic_outputs, axis=-1)

    def _apply_networks(self, obs, obs_critic, agent_id):
        """
        Apply actor and critic networks using hypernetwork-generated weights.

        Args:
            obs: Agent observations
            obs_critic: Global observations for critic
            agent_id: Agent IDs (integers) used to select pre-generated weights

        Returns:
            means: Action means from actor network
            log_stds: Log standard deviations from actor network
            critic_outputs: Value estimates from critic network

        Note:
            - self.agent_embeddings is a one-hot matrix (num_agents, num_agents) when
              USE_AGENT_ID_EMBEDDINGS=False, or learned embeddings otherwise
            - The hypernetwork receives the entire embedding matrix and generates
              weights for ALL agents simultaneously
            - agent_id is used to index and select the appropriate agent's weights
        """
        batch_size = obs.shape[0]

        # Pre-compute all hypernetwork outputs for all agents using one-hot embeddings
        # actor_weights/biases shape: (num_agents, ...) - one set per agent
        actor_weights, actor_biases = self.actor_hypernet(self.agent_embeddings)
        critic_weights, critic_biases = self.critic_hypernet(self.agent_embeddings)

        def apply_hypernet(obs, weights, biases):
            """
            Apply a network with hypernetwork-generated weights.

            Args:
                obs: Input observation
                weights: List of weight matrices
                biases: List of bias vectors

            Returns:
                Network output
            """
            x = obs
            for w, b in zip(weights[:-1], biases[:-1]):
                x = self.activation_fn(jnp.matmul(x, w.reshape(x.shape[-1], -1)) + b)
            return jnp.matmul(x, weights[-1].reshape(x.shape[-1], -1)) + biases[-1]

        # Vectorize apply_hypernet over all agents
        vmap_apply_hypernet = jax.vmap(apply_hypernet, in_axes=(None, 0, 0))

        # Apply networks for all agents simultaneously
        # Shape: (num_agents, batch_size, action_dim * 2) for actor (means + log_stds)
        # Shape: (num_agents, batch_size, 1) for critic
        actor_outputs_all = vmap_apply_hypernet(obs, actor_weights, actor_biases)
        critic_outputs_all = vmap_apply_hypernet(
            obs_critic, critic_weights, critic_biases
        )

        # Select outputs for each agent based on agent_id
        batch_indices = jnp.arange(batch_size)
        actor_outputs = actor_outputs_all[agent_id, batch_indices]
        critic_outputs = critic_outputs_all[agent_id, batch_indices]

        if self.action_space_type == "continuous":
            # Split actor outputs into means and log_stds for continuous actions
            means, log_stds = jnp.split(actor_outputs, 2, axis=-1)
            # Clip log_stds for numerical stability
            # Upper bound of -0.3 gives max std ≈ 0.74, balanced exploration
            log_stds = jnp.clip(log_stds, -20, -0.3)
        else:
            means = actor_outputs  # logits for categorical policy
            log_stds = jnp.zeros_like(actor_outputs)

        return means, log_stds, critic_outputs


class TransitionInfo(NamedTuple):
    """
    Information about completed episodes during transitions.

    Attributes:
        returned_episode_returns: Returns of completed episodes
        returned_episode_lengths: Lengths of completed episodes
    """

    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@dataclass
class Transition:
    """
    Stores a single step transition for training.

    Contains all necessary information for PPO updates.

    Attributes:
        done: Done flags for each agent
        action: Actions taken by each agent
        value: Value estimates
        reward: Rewards received
        log_prob: Log probabilities of actions
        obs: Observations (local)
        global_obs: Global observations (all agents)
        info: Additional episode information
    """

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    info: TransitionInfo


def initialize_metrics_storage(config, metric_keys):
    """
    Initialize storage for training metrics.

    Args:
        config: Configuration dictionary
        metric_keys: Keys of metrics to track

    Returns:
        Dictionary mapping metric names to arrays
    """
    return {k: np.zeros(config["NUM_UPDATES"]) for k in metric_keys}


@jax.jit
def update_metrics(metrics: dict, new_values: dict, update_idx: int) -> dict:
    """
    Update metrics with new values at specified index.

    Args:
        metrics: Current metrics dictionary
        new_values: New values to insert
        update_idx: Index to update

    Returns:
        Updated metrics dictionary
    """

    def _update(arr, val):
        if isinstance(arr, jnp.ndarray):
            return arr.at[update_idx].set(val)
        elif isinstance(val, dict):
            return jax.tree_map(lambda a, v: _update(a, v), arr, val)
        return val

    return jax.tree_map(lambda x, y: _update(x, y), metrics, new_values)


def make_train(config):
    """
    Create the training function based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        train: Function that performs training
    """
    # Verify one-hot encoding is enabled for agent IDs
    assert config["env"]["ENV_KWARGS"].get(
        "one_hot_encode_agent_id"
    ), "one_hot_ids must be True"

    # Initialize environment with action-space consistency
    action_space_type = config.get("ACTION_SPACE_TYPE", "continuous")
    train_env_kwargs = dict(config["TRAIN_ENV_KWARGS"])
    train_env_kwargs["continuous_actions"] = action_space_type != "discrete"

    env, possible_agents, action_dim, num_actions, observation_size = make_env(
        config["ENV_NAME"],
        num_envs=config["NUM_ENVS"],
        **train_env_kwargs,
    )

    # Calculate derived parameters
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        """Linear learning rate decay schedule."""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, wb_run=None):
        """
        Main training function.

        Args:
            rng: JAX random number generator key
            wb_run: Weights & Biases run object

        Returns:
            Dictionary of training results and environment
        """
        # Observation size (no agent IDs in observations)
        true_obs_size = observation_size
        critic_obs_size = true_obs_size * env.num_agents
        is_discrete_action_space = action_space_type == "discrete"

        # Initialize network
        network = ActorCritic(
            action_dim,
            activation=config["ACTIVATION"],
            actor_layers=config.get("ACTOR_LAYERS"),
            critic_layers=config.get("CRITIC_LAYERS"),
            num_agents=env.num_agents,
            embedding_dim=config.get("HYPERNET_EMBEDDING_DIM", None),
            use_agent_id_embeddings=config.get("USE_AGENT_ID_EMBEDDINGS", False),
            init_scale=config.get("INIT_SCALE", np.sqrt(2)),
            use_bias_in_hypernet=config.get("USE_BIAS_IN_HYPERNET", True),
            observation_dim=true_obs_size,
            critic_obs_size=critic_obs_size,
            hypernet_hidden_dims=config.get("HYPERNET_HIDDEN_DIMS", (64,)),
            action_space_type=config.get("ACTION_SPACE_TYPE", "continuous"),
        )

        # Initialize network parameters
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, observation_size))
        num_agents = len(env.agents)
        init_x_critic = jnp.zeros((1, critic_obs_size))
        network_params = network.init(_rng, init_x, init_x_critic)

        # Configure optimizer with optional learning rate annealing
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # Create train state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # JIT-compile the network apply function
        network.apply = jax.jit(network.apply)

        # Load checkpoint if specified
        zero_shot_eval = False
        if config.get("CHECKPOINT_LOAD_DIR") is not None:
            print(f"Loading from {config.get('CHECKPOINT_LOAD_DIR')}")
            load_checkpointer = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename="checkpoints")
            )
            loaded_checkpoints = load_checkpointer.restore(
                config.get("CHECKPOINT_LOAD_DIR"), item=train_state.params
            )
            train_state = TrainState.create(
                apply_fn=train_state.apply_fn, params=loaded_checkpoints, tx=tx
            )
            zero_shot_eval = True

        # Initialize episode statistics
        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
            returned_episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            returned_episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
        )

        # Preallocate arrays for transitions (more efficient memory usage)
        transitions = Transition(
            done=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=bool
            ),
            action=(
                np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.int32,
                )
                if is_discrete_action_space
                else np.zeros(
                    (
                        config["NUM_STEPS"],
                        config["NUM_ENVS"] * num_agents,
                        action_dim,
                    ),
                    dtype=jnp.float32,
                )
            ),
            value=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            reward=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            log_prob=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            obs=np.zeros(
                (
                    config["NUM_STEPS"],
                    config["NUM_ENVS"] * num_agents,
                    observation_size,
                ),
                dtype=jnp.float32,
            ),
            global_obs=np.zeros(
                (
                    config["NUM_STEPS"],
                    config["NUM_ENVS"] * num_agents,
                    critic_obs_size,
                ),
                dtype=jnp.float32,
            ),
            info=TransitionInfo(
                returned_episode_returns=np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.float32,
                ),
                returned_episode_lengths=np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.int32,
                ),
            ),
        )

        def dict_obs_to_array(obs_dict):
            """
            Convert dict observations to flat array.

            Args:
                obs_dict: Dictionary of observations {agent_0: array, agent_1: array, ...}

            Returns:
                Flat array of shape (num_envs * num_agents, obs_dim)
            """
            if isinstance(obs_dict, dict):
                # Stack observations in agent order
                obs_list = [obs_dict[agent] for agent in env.agents]
                # Shape: (num_agents, num_envs, obs_dim) -> (num_envs, num_agents, obs_dim)
                obs_array = np.array(obs_list).transpose(1, 0, 2)
                # Flatten to (num_envs * num_agents, obs_dim)
                return obs_array.reshape(-1, obs_array.shape[-1])
            else:
                # Already an array
                return obs_dict

        def dict_values_to_array(value_dict):
            """
            Convert dict of values (rewards, dones) to flat array.

            Args:
                value_dict: Dictionary of values {agent_0: array, agent_1: array, ...}

            Returns:
                Flat array of shape (num_envs * num_agents,) or (num_envs,) for shared values
            """
            if isinstance(value_dict, dict):
                # Stack values in agent order
                value_list = [value_dict[agent] for agent in env.agents]
                # Shape: (num_agents, num_envs) -> (num_envs, num_agents)
                value_array = np.array(value_list).T
                # Flatten to (num_envs * num_agents,)
                return value_array.flatten()
            else:
                # Already an array
                return value_dict

        @jax.jit
        def concat_local_obs_into_global_obs(obs):
            """
            Create global observations by concatenating all agents' observations.

            Args:
                obs: Local observations (no agent IDs)

            Returns:
                Global observations for each agent
            """
            # Reshape observations: (num_envs * num_agents, obs_dim) -> (num_envs, num_agents, obs_dim)
            reshaped_obs = obs.reshape((config["NUM_ENVS"], num_agents, -1))

            obs_dim = reshaped_obs.shape[-1]
            global_obs = jnp.zeros(
                (config["NUM_ENVS"] * num_agents, num_agents * obs_dim)
            )

            # For each environment, create the global observation
            for i in range(config["NUM_ENVS"]):
                flat_global_obs_per_env = reshaped_obs[i].flatten()
                env_begin = i * num_agents
                env_end = env_begin + num_agents
                global_obs = global_obs.at[env_begin:env_end].set(
                    jnp.tile(flat_global_obs_per_env, (num_agents,)).reshape(
                        num_agents, num_agents * obs_dim
                    )
                )
            return global_obs

        # Initialize environment
        rng, reset_rng = jax.random.split(rng)
        int_seed = jax.random.randint(
            reset_rng, shape=(1,), minval=1, maxval=1000000
        ).item()
        obsv, infos = env.reset(seed=int_seed)
        obsv = dict_obs_to_array(obsv)  # Convert dict to array
        global_obsv = concat_local_obs_into_global_obs(obsv)
        env_state = {}
        env_step = env.step

        def step_env_wrapped(action: Any) -> Any:
            """
            Step the environment and process the results.

            Args:
                action: Actions to take

            Returns:
                next_obs: Next observations
                reward: Rewards received
                termination: Terminal states
                truncs: Truncation flags
                info: Additional info
                global_obs: Global observations
            """
            # Keep action dimensionality intact for continuous control
            # (shape is typically [num_envs * num_agents, action_dim]).
            next_obs, reward, termination, truncs, info = env_step(action)
            # Convert dict outputs to arrays
            next_obs = dict_obs_to_array(next_obs)
            reward = dict_values_to_array(reward)
            reward = np.asarray(reward, dtype=np.float32)
            reward = reward * float(config.get("REWARD_SCALE", 1.0))
            reward_clip = config.get("REWARD_CLIP", None)
            if reward_clip is not None:
                reward = np.clip(reward, -float(reward_clip), float(reward_clip))
                reward = np.nan_to_num(
                    reward,
                    nan=0.0,
                    posinf=float(reward_clip),
                    neginf=-float(reward_clip),
                )
            else:
                reward = np.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
            termination = dict_values_to_array(termination)
            truncs = dict_values_to_array(truncs)
            global_obs = concat_local_obs_into_global_obs(next_obs)
            return next_obs, reward, termination, truncs, info, global_obs

        @jax.jit
        def update_episode_stats(episode_stats, reward, done):
            """
            Update episode statistics after a step.

            Args:
                episode_stats: Current episode statistics
                reward: Rewards received
                done: Done flags

            Returns:
                Updated episode statistics
            """
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            returned_episode_returns = jnp.where(
                done, new_episode_return, episode_stats.returned_episode_returns
            )
            returned_episode_lengths = jnp.where(
                done, new_episode_length, episode_stats.returned_episode_lengths
            )

            return episode_stats.replace(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=returned_episode_returns,
                returned_episode_lengths=returned_episode_lengths,
            )

        @jax.jit
        def _select_action(params, obs, obs_critic, _rng):
            """
            Select actions for all agents using continuous action distributions.

            Args:
                params: Network parameters
                obs: Agent observations
                obs_critic: Global observations
                _rng: Random number generator key

            Returns:
                action_for_training: Actions used for log-prob/loss computation
                action_for_env: Actions executed in the environment
                log_prob: Log probabilities of actions
                value: Value function estimates
            """
            means_or_logits, log_stds, value = network.apply(params, obs, obs_critic)
            if is_discrete_action_space:
                pi = distrax.Categorical(logits=means_or_logits)
                action_for_training = pi.sample(seed=_rng).astype(jnp.int32)
                log_prob = pi.log_prob(action_for_training)
                action_for_env = action_for_training
            else:
                stds = jnp.exp(log_stds)
                pi = distrax.MultivariateNormalDiag(
                    loc=means_or_logits, scale_diag=stds
                )
                action_for_training = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action_for_training)
                # Clip actions to valid range (VMAS typically expects [-1, 1])
                action_for_env = jnp.clip(action_for_training, -1.0, 1.0)
            return action_for_training, action_for_env, log_prob, value

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            """
            Select deterministic actions for evaluation.

            Args:
                params: Network parameters
                obs: Observations
                _rng: Random number generator key

            Returns:
                actions: Selected actions
            """
            # Create dummy global observations for critic
            dummy_obs_critic = jnp.zeros((obs.shape[0], critic_obs_size))
            means_or_logits, log_stds, _ = network.apply(params, obs, dummy_obs_critic)

            # Use either stochastic or deterministic actions based on config
            if is_discrete_action_space:
                if config.get("eval_stochastic"):
                    pi = distrax.Categorical(logits=means_or_logits)
                    action = pi.sample(seed=_rng)
                else:
                    action = jnp.argmax(means_or_logits, axis=-1)
                action = action.astype(jnp.int32)
            else:
                if config.get("eval_stochastic"):
                    stds = jnp.exp(log_stds)
                    pi = distrax.MultivariateNormalDiag(
                        loc=means_or_logits, scale_diag=stds
                    )
                    action = pi.sample(seed=_rng)
                else:
                    # Use mean as deterministic action
                    action = means_or_logits

                # Clip actions to valid range
                action = jnp.clip(action, -1.0, 1.0)

            return (action,)

        # Get PPO update function
        calculate_advantage_and_update_ppo = get_update_fn(config, network)

        def _update_step(runner_state, unused):
            """
            Perform a single training update step.

            Args:
                runner_state: Current runner state
                unused: Unused parameter for JAX compatibility

            Returns:
                Updated runner state and metrics
            """
            train_state, env_state, last_obs, last_global_obs, rng, episode_stats = (
                runner_state
            )

            # Collect transitions
            for t in range(config["NUM_STEPS"]):
                # Select actions
                rng, _rng = jax.random.split(rng)
                action_for_training, action_for_env, log_prob, value = _select_action(
                    train_state.params, last_obs, last_global_obs, _rng
                )

                # Step environment
                np_action = np.array(action_for_env)
                obsv, reward, termination, truncs, info, global_obs = step_env_wrapped(
                    np_action
                )

                # Update episode statistics
                done = np.logical_or(termination, truncs)
                episode_stats = update_episode_stats(episode_stats, reward, done)

                # Store transitions
                transitions.done[t] = done
                transitions.action[t] = action_for_training
                transitions.value[t] = value
                transitions.reward[t] = reward
                transitions.log_prob[t] = log_prob
                transitions.obs[t] = last_obs
                transitions.global_obs[t] = last_global_obs
                transitions.info.returned_episode_returns[t] = (
                    episode_stats.returned_episode_returns
                )
                transitions.info.returned_episode_lengths[t] = (
                    episode_stats.returned_episode_lengths
                )

                # Update observations
                last_obs = obsv
                last_global_obs = global_obs

            # Update policy
            update_state, loss_info = calculate_advantage_and_update_ppo(
                transitions, last_obs, last_global_obs, train_state, rng
            )
            train_state = update_state[0]

            # Collect metrics
            metric = {
                "returned_episode_returns": episode_stats.returned_episode_returns,
                "returned_episode_lengths": episode_stats.returned_episode_lengths,
            }
            rng = update_state[-1]

            # Average metrics
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info}

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_global_obs,
                rng,
                episode_stats,
            )
            return runner_state, metric

        # Initialize runner state
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, global_obsv, _rng, episode_stats)

        # Set up checkpointer
        checkpointers = checkpointer.Checkpointer(
            PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
        )

        # Perform zero-shot evaluation if loading from checkpoint
        if zero_shot_eval:
            print("Running zero shot eval")
            # Control video capture based on number of evaluation episodes
            if config.get("EVAL_EPISODES") <= 40:
                capture_video = True
            else:
                capture_video = False

            use_custom_eval = (
                config.get("USE_CUSTOM_EVAL", False) and CUSTOM_EVAL_AVAILABLE
            )

            if use_custom_eval:
                # Use custom evaluation with detailed metrics
                print("Using custom evaluation for zero-shot...")
                eval_data = evaluate_policy_with_metrics(
                    env=env,
                    network=network,
                    params=runner_state[0].params,
                    num_eval_episodes=max(
                        1, config["EVAL_EPISODES"] // config["NUM_ENVS"]
                    ),
                    num_envs=config["NUM_ENVS"],
                    max_eval_steps=config.get("MAX_EVAL_STEPS", 128),
                    seed=42,
                    deterministic=config.get("EVAL_DETERMINISTIC", True),
                    scenario_name=config["ENV_NAME"],
                )

                # Log and print results
                if wb_run is not None:
                    log_dict = {
                        "eval/completion_rate": eval_data["completion_rate"],
                        "eval/avg_reward": eval_data["avg_reward"],
                        "eval/std_reward": eval_data["std_reward"],
                        "eval/avg_episode_length": eval_data["avg_episode_length"],
                    }
                    if "food_found_percentage" in eval_data:
                        log_dict["eval/food_found_percentage"] = eval_data[
                            "food_found_percentage"
                        ]
                    wb_run.log(log_dict, step=0)

                print(
                    f"  Zero-shot Completion Rate: {eval_data['completion_rate']:.2f}%"
                )
                print(f"  Zero-shot Average Reward: {eval_data['avg_reward']:.4f}")
                if "food_found_percentage" in eval_data:
                    print(
                        f"  Zero-shot Food Found: {eval_data['food_found_percentage']:.2f}%"
                    )

            elif BASELINES_AVAILABLE and run_eval_jax is not None:
                eval_data = run_eval_jax(
                    cfg=config,
                    agent_state=runner_state[0].params,
                    writer=wb_run,
                    acting_fns=_select_action_eval,
                    eval_seed=42,
                    global_step=0,
                    capture_video=capture_video,
                    recurrent=False,
                    shared_weights=True,
                    parallel=config.get("EVAL_PARALLEL", True),
                )
            else:
                print("Warning: No evaluation method available for zero-shot eval.")
                eval_data = {}

        # Initialize training metrics
        training_metrics = None

        # Calculate team diversity metrics if requested
        calculate_team_diversity_metrics = (
            config.get("CALCULATE_TEAM_DIVERSITY_METRICS", False)
            and BASELINES_AVAILABLE
        )
        if calculate_team_diversity_metrics:
            _keys = jax.random.PRNGKey(config["SEED"])
            calculate_team_diversity(
                network,
                runner_state[0],
                _keys,
                num_agents,
                centralised_critic=True,
                param_sharing=True,
            )

        # Initialize evaluation metrics and training variables
        eval_metrics = []
        start_time = time.time()
        global_step = 0
        next_eval_step = config["EVAL_INTERVAL"]
        next_capture_video_step = config.get("CAPTURE_VIDEO_INTERVAL", None)
        next_checkpoint_step = config.get("CHECKPOINT_INTERVAL", None)

        # Early return if only evaluating
        eval_only = config.get("EVAL_ONLY", False)
        if eval_only:
            return {"metrics": {}, "eval_metrics": [(0, eval_data)]}, env

        # Main training loop
        for update in range(config["NUM_UPDATES"]):
            final_update = update == config["NUM_UPDATES"] - 1
            update_time_start = time.time()

            # Perform update
            runner_state, ret_metric = _update_step(runner_state, None)
            global_step += config["NUM_STEPS"] * config["NUM_ENVS"]

            # Log training speed periodically
            if update % 100 == 0:
                print(f"Update: {update}/{config['NUM_UPDATES']}")
                sps = int(global_step / (time.time() - start_time))
                sps_update = int(
                    config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                    / (time.time() - update_time_start)
                )
                print("SPS:", sps, sps_update)
                wb_run.log({"charts/SPS": sps}, global_step)
                wb_run.log({"charts/SPS_update": sps_update}, global_step)

            # Initialize or update metrics
            if training_metrics is None:
                training_metrics = initialize_metrics_storage(config, ret_metric.keys())
            training_metrics = update_metrics(training_metrics, ret_metric, update)

            # Log training metrics to wandb
            if wb_run is not None:
                train_log_dict = {}
                # Log loss metrics
                if "actor_loss" in ret_metric:
                    train_log_dict["train/actor_loss"] = float(ret_metric["actor_loss"])
                if "critic_loss" in ret_metric:
                    train_log_dict["train/critic_loss"] = float(
                        ret_metric["critic_loss"]
                    )
                if "total_loss" in ret_metric:
                    train_log_dict["train/total_loss"] = float(ret_metric["total_loss"])
                if "entropy" in ret_metric:
                    train_log_dict["train/entropy"] = float(ret_metric["entropy"])
                if "ratio" in ret_metric:
                    train_log_dict["train/ratio"] = float(ret_metric["ratio"].mean())
                # Log episode metrics
                if "returned_episode_returns" in ret_metric:
                    train_log_dict["train/episode_return"] = float(
                        ret_metric["returned_episode_returns"]
                    )
                if "returned_episode_lengths" in ret_metric:
                    train_log_dict["train/episode_length"] = float(
                        ret_metric["returned_episode_lengths"]
                    )

                if train_log_dict:
                    wb_run.log(train_log_dict, step=global_step)

                # Log football-specific metrics (per-rollout goals and rolling win rate)
                if hasattr(env, "get_win_rate"):
                    football_stats = env.get_win_rate()
                    # Compute per-rollout deltas
                    prev = getattr(
                        env,
                        "_prev_football_stats",
                        {"blue_goals": 0, "red_goals": 0, "total_episodes": 0},
                    )
                    rollout_blue = football_stats["blue_goals"] - prev["blue_goals"]
                    rollout_red = football_stats["red_goals"] - prev["red_goals"]
                    rollout_episodes = (
                        football_stats["total_episodes"] - prev["total_episodes"]
                    )
                    env._prev_football_stats = dict(football_stats)
                    wb_run.log(
                        {
                            "football/blue_goals": rollout_blue,
                            "football/red_goals": rollout_red,
                            "football/win_rate": football_stats["win_rate"],
                            "football/episodes": rollout_episodes,
                        },
                        step=global_step,
                    )

            # Run evaluation periodically
            record_final_episode = config.get("CAPTURE_VIDEO_INTERVAL") and final_update
            if (global_step >= next_eval_step) or record_final_episode:
                if (
                    next_capture_video_step and global_step >= next_capture_video_step
                ) or record_final_episode:
                    next_capture_video_step += config["CAPTURE_VIDEO_INTERVAL"]
                    capture_video = True
                else:
                    capture_video = False

                # Choose evaluation method based on config
                use_custom_eval = (
                    config.get("USE_CUSTOM_EVAL", False) and CUSTOM_EVAL_AVAILABLE
                )

                if use_custom_eval:
                    # Use custom evaluation with detailed metrics
                    print(f"\nRunning custom evaluation at step {global_step}...")
                    eval_metrics_custom = evaluate_policy_with_metrics(
                        env=env,
                        network=network,
                        params=runner_state[0].params,
                        num_eval_episodes=max(
                            1, config["EVAL_EPISODES"] // config["NUM_ENVS"]
                        ),
                        num_envs=config["NUM_ENVS"],
                        max_eval_steps=config.get("MAX_EVAL_STEPS", 128),
                        seed=42 + update,
                        deterministic=config.get("EVAL_DETERMINISTIC", True),
                        scenario_name=config["ENV_NAME"],
                    )

                    # Log custom metrics to wandb
                    if wb_run is not None:
                        log_dict = {
                            "eval/completion_rate": eval_metrics_custom[
                                "completion_rate"
                            ],
                            "eval/avg_reward": eval_metrics_custom["avg_reward"],
                            "eval/std_reward": eval_metrics_custom["std_reward"],
                            "eval/avg_episode_length": eval_metrics_custom[
                                "avg_episode_length"
                            ],
                        }
                        if "food_found_percentage" in eval_metrics_custom:
                            log_dict["eval/food_found_percentage"] = (
                                eval_metrics_custom["food_found_percentage"]
                            )
                            log_dict["eval/avg_food_per_episode"] = eval_metrics_custom[
                                "avg_food_per_episode"
                            ]
                        wb_run.log(log_dict, step=global_step)

                        # Log football eval stats
                        if hasattr(env, "get_win_rate"):
                            # Stats accumulated since last reset include the eval rollouts
                            fs = env.get_win_rate()
                            wb_run.log(
                                {
                                    "eval/football_blue_goals": fs["blue_goals"],
                                    "eval/football_red_goals": fs["red_goals"],
                                    "eval/football_win_rate": fs["win_rate"],
                                },
                                step=global_step,
                            )

                    # Print metrics
                    print(
                        f"  Completion Rate: {eval_metrics_custom['completion_rate']:.2f}%"
                    )
                    print(
                        f"  Average Reward: {eval_metrics_custom['avg_reward']:.4f} ± {eval_metrics_custom['std_reward']:.4f}"
                    )
                    if "food_found_percentage" in eval_metrics_custom:
                        print(
                            f"  Food Found: {eval_metrics_custom['food_found_percentage']:.2f}%"
                        )
                    print(
                        f"  Avg Episode Length: {eval_metrics_custom['avg_episode_length']:.2f}"
                    )

                    eval_data = eval_metrics_custom

                elif BASELINES_AVAILABLE and run_eval_jax is not None:
                    # Use standard evaluation from baselines
                    eval_data = run_eval_jax(
                        cfg=config,
                        agent_state=runner_state[0].params,
                        writer=wb_run,
                        acting_fns=_select_action_eval,
                        eval_seed=42,
                        global_step=global_step,
                        capture_video=capture_video,
                        recurrent=False,
                        shared_weights=True,
                        parallel=config.get("EVAL_PARALLEL", True),
                    )
                else:
                    print("Warning: No evaluation method available.")
                    eval_data = {}

                eval_metrics.append((global_step, eval_data))
                next_eval_step += config["EVAL_INTERVAL"]

            # Extract and save agent embeddings periodically
            if BASELINES_AVAILABLE and (
                update % config.get("SAVE_EMBEDDINGS_INTERVAL", 5000) == 0
                or final_update
            ):
                extract_and_save_agent_embeddings(
                    train_state.params, global_step, config, wb_run
                )

            # Save checkpoints periodically
            if (final_update) or (
                next_checkpoint_step and global_step >= next_checkpoint_step
            ):
                agent_identity = f"agent_{config['SEED']}_seed"
                model_path = f"{config['CHP_DIR']}/{config['EXP_NAME']}_{global_step}_steps_{update}_updates.{agent_identity}"
                save_args = orbax_utils.save_args_from_target(runner_state[0].params)
                checkpointers.save(
                    model_path, runner_state[0].params, save_args=save_args
                )
                print(f"Model saved to {model_path} at step {global_step}")
                next_checkpoint_step += config["CHECKPOINT_INTERVAL"]

        return {
            "runner_state": runner_state,
            "metrics": training_metrics,
            "eval_metrics": eval_metrics,
        }, env

    return train


def load_config(config_name):
    """
    Load and merge YAML configurations.

    Args:
        config_name: Name of the main config file (without .yaml extension)

    Returns:
        Merged configuration dictionary
    """
    import re

    config_path = Path(config_name + ".yaml")

    # Load main config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle defaults (config composition)
    if "defaults" in config:
        merged_config = {}
        for default_config in config["defaults"]:
            if default_config == "_self_":
                continue
            # Load the referenced config and nest it under "env" key (like Hydra does)
            default_path = Path(default_config + ".yaml")
            if default_path.exists():
                with open(default_path, "r") as f:
                    default_cfg = yaml.safe_load(f)
                    # Nest environment config under "env" key
                    merged_config["env"] = default_cfg

        # Merge main config on top (excluding defaults)
        for k, v in config.items():
            if k != "defaults":
                merged_config[k] = v
        config = merged_config

    # Resolve variable interpolations (multiple passes for dependencies)
    def resolve_value(value, root_cfg, config_name):
        """Resolve a single interpolation string."""
        if not isinstance(value, str) or "${" not in value:
            return value

        original = value
        # Handle ${env.ENV_NAME}
        if "env" in root_cfg and "ENV_NAME" in root_cfg["env"]:
            value = value.replace("${env.ENV_NAME}", root_cfg["env"]["ENV_NAME"])

        # Handle ${hydra:runtime.config_name}
        value = value.replace("${hydra:runtime.config_name}", config_name)

        # Handle ${now:%Y-%m-%d_%H-%M-%S}
        if "${now:" in value:
            match = re.search(r"\$\{now:([^}]+)\}", value)
            if match:
                timestamp = datetime.now().strftime(match.group(1))
                value = re.sub(r"\$\{now:[^}]+\}", timestamp, value)

        # Handle simple variable references like ${ALG}, ${ENV_NAME}, etc.
        max_iterations = 10
        iteration = 0
        while "${" in value and iteration < max_iterations:
            iteration += 1
            match = re.search(r"\$\{([^}:]+)\}", value)
            if match:
                var_name = match.group(1)
                # Try to find in root config
                if var_name in root_cfg:
                    replacement = str(root_cfg[var_name])
                    value = value.replace("${" + var_name + "}", replacement)
                else:
                    break
            else:
                break

        return value

    def resolve_interpolations(cfg, root_cfg, config_name):
        """Recursively resolve all interpolations in the config."""
        if isinstance(cfg, dict):
            for key, value in list(cfg.items()):
                if isinstance(value, str):
                    cfg[key] = resolve_value(value, root_cfg, config_name)
                elif isinstance(value, (dict, list)):
                    resolve_interpolations(value, root_cfg, config_name)
        elif isinstance(cfg, list):
            for i, item in enumerate(cfg):
                if isinstance(item, str):
                    cfg[i] = resolve_value(item, root_cfg, config_name)
                elif isinstance(item, (dict, list)):
                    resolve_interpolations(item, root_cfg, config_name)

    # Multiple passes to handle dependencies
    for _ in range(3):
        resolve_interpolations(config, config, config_name)

    # Convert numeric types that may have been stringified
    def convert_types(cfg):
        """Convert string representations of numbers back to numeric types."""
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if isinstance(value, str):
                    # Try to convert to float/int
                    try:
                        # Check if it's a number in scientific notation or regular number
                        if "e" in value.lower() or "." in value:
                            cfg[key] = float(value)
                        else:
                            # Try int first
                            cfg[key] = int(value)
                    except (ValueError, AttributeError):
                        # Keep as string if conversion fails
                        pass
                elif isinstance(value, (dict, list)):
                    convert_types(value)
        elif isinstance(cfg, list):
            for i, item in enumerate(cfg):
                if isinstance(item, str):
                    try:
                        if "e" in item.lower() or "." in item:
                            cfg[i] = float(item)
                        else:
                            cfg[i] = int(item)
                    except (ValueError, AttributeError):
                        pass
                elif isinstance(item, (dict, list)):
                    convert_types(item)

    convert_types(config)

    return config


def main(config):
    """
    Main entry point for training.

    Args:
        config: Configuration dictionary
    """
    print("Starting training")

    # Initialize wandb
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["EXP_TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=config["RUN_NAME"],
        save_code=True,
        reinit=True,
        group=config["GROUP"],
    )

    # Set up checkpoint directory
    config["CHP_DIR"] = f"{wandb.run.dir}/models/{config['RUN_NAME']}"  # type: ignore

    # Load checkpoint if specified
    if config.get("CHECKPOINT_NAME") is not None:
        artifact = run.use_artifact(config.get("CHECKPOINT_NAME"), type="checkpoint")
        artifact_dir = artifact.download()
        checkpoint_load_dir = f"{artifact_dir}/{config.get('CHECKPOINT_FOLDER')}"

        config["CHECKPOINT_LOAD_DIR"] = checkpoint_load_dir
        print(f"checkpoint_load_dir: {checkpoint_load_dir}")

    # Set up RNG
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Check if there's a nested 'env' key (from Hydra config merging)
    env_config = config["env"]["env"] if "env" in config["env"] else config["env"]

    # Add environment kwargs to config
    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])

    # Also pass capability configuration from env_config
    if "use_fixed_capabilities" in env_config:
        config["TRAIN_ENV_KWARGS"]["use_fixed_capabilities"] = env_config[
            "use_fixed_capabilities"
        ]
        config["TEST_ENV_KWARGS"]["use_fixed_capabilities"] = env_config[
            "use_fixed_capabilities"
        ]
    if "fixed_capabilities" in env_config:
        config["TRAIN_ENV_KWARGS"]["fixed_capabilities"] = env_config[
            "fixed_capabilities"
        ]
        config["TEST_ENV_KWARGS"]["fixed_capabilities"] = env_config[
            "fixed_capabilities"
        ]

    # Run training
    train = make_train(config)
    out, env = train(rngs[0], run)

    # Log metrics
    if BASELINES_AVAILABLE:
        log_train_metrics(config, out["metrics"], run)
        log_eval_metrics(config, out["eval_metrics"], run)
    else:
        print("\nTraining completed. Logging metrics...")
        # Basic local logging if baselines not available
        if out.get("eval_metrics"):
            print(f"\nFinal Evaluation Metrics:")
            for step, metrics in out["eval_metrics"][-5:]:  # Show last 5 evaluations
                print(f"  Step {step}:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.4f}")

    # Clean up
    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HyperMARL MAPPO")
    parser.add_argument(
        "--config-name",
        type=str,
        default="config_hypermarl_dispersion",
        help="Name of the config file (without .yaml extension)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config file)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_name)

    # Override seed if provided
    if args.seed is not None:
        config["SEED"] = args.seed
        print(f"Using seed override: {args.seed}")

    # Run training
    main(config)
