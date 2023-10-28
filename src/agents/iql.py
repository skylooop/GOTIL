"""Implementations of algorithms for continuous control."""
import dataclasses
import functools
from dataclasses import dataclass, asdict

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ensemblize, MLP

import pyrallis
from tqdm.auto import tqdm
import gym
import d4rl

import wandb
import flax
import ml_collections

@dataclass
class TrainConfig:
    project: str = "CORL"
    group: str = "IQL-Flax"
    name: str = "IQL-Flax"
    dataset_id: str = "antmaze-large-diverse-v2"
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 10.0
    iql_tau: float = 0.9 #experctile
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 10_000_000
    # training batch size
    batch_size: int = 512
    # whether to normalize states
    normalize_state: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = True
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5_000)
    # number of episodes to run during evaluation
    n_episodes: int = 100
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # training random seed
    seed: int = 0

def expectile_loss(diff, expectile=0.9):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    value: TrainState
    target_value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def critic_loss(agent, batch, critic_params):
        next_v = agent.target_value(batch['next_observations'])
        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v
        q1, q2 = agent.critic(batch['observations'], batch['actions'], params=critic_params)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
        }
    
    def value_loss(agent, batch, value_params):
        q1, q2 = agent.critic(batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = agent.value(batch['observations'], params=value_params)
        value_loss = expectile_loss(q-v, agent.config['expectile']).mean()
        advantage = q - v
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'abs adv mean': jnp.abs(advantage).mean(),
            'adv mean': advantage.mean(),
            'adv max': advantage.max(),
            'adv min': advantage.min(),
        }

    def actor_loss(agent, batch, actor_params):
        v = agent.value(batch['observations'])
        q1, q2 = agent.critic(batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * agent.config['temperature'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = agent.actor(batch['observations'], params=actor_params)
        log_probs = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_probs).mean()

        sorted_adv = jnp.sort(q-v)[::-1]
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': q - v,
            'bc_log_probs': log_probs.mean(),
            'adv median': jnp.median(q - v),
            'adv top 1%': sorted_adv[int(len(sorted_adv) * 0.01)],
            'adv top 10%': sorted_adv[int(len(sorted_adv) * 0.1)],
            'adv top 25%': sorted_adv[int(len(sorted_adv) * 0.25)],
            'adv top 25%': sorted_adv[int(len(sorted_adv) * 0.25)],
            'adv top 75%': sorted_adv[int(len(sorted_adv) * 0.75)],
        }        

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        def critic_loss_fn(critic_params):
            return agent.critic_loss(batch, critic_params)
        
        def value_loss_fn(value_params):
            return agent.value_loss(batch, value_params)

        def actor_loss_fn(actor_params):
            return agent.actor_loss(batch, actor_params)

        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_value = target_update(agent.value, agent.target_value, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return agent.replace(critic=new_critic, target_value=new_target_value, value=new_value, actor=new_actor), {
            **critic_info, **value_info, **actor_info
        }


def create_learner(
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 value_def,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 value_tx=None,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = 1e6,
                 opt_decay_schedule: str = "cosine",
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims, action_dim=action_dim, 
            log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=actor_lr)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        critic_def = ensemblize(Critic, num_qs=2)(hidden_dims)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
        
        value_params = value_def.init(value_key, observations)['params']
        if value_tx is None:
            value_tx = optax.adam(learning_rate=value_lr)
        value = TrainState.create(value_def, value_params, tx=value_tx)
        target_value = TrainState.create(value_def, value_params)

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, expectile=expectile, target_update_rate=tau, 
        ))

        return IQLAgent(rng, critic=critic, value=value, target_value=target_value, actor=actor, config=config)

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    
    def normalize_state(state):
        return (state - state_mean) / state_std
    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std
    

class ReplayBuffer(flax.struct.PyTreeNode):
    data: Dict[str, jax.Array]
    
    @classmethod
    def create_from_d4rl(cls, env, normalize_reward=True, normalize_state=True) -> "ReplayBuffer":
        dataset = d4rl.qlearning_dataset(env)
        if normalize_reward:
            dataset['rewards'] -= 1.0
        if normalize_state:
            state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        else:
            state_mean, state_std = 0, 1
            
        dataset["observations"] = normalize_states(
            dataset["observations"], state_mean, state_std
        )
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )
        buffer = {
            "observations": jnp.asarray(dataset["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(dataset["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(dataset["rewards"], dtype=jnp.float32),
            "next_observations": jnp.asarray(dataset["next_observations"], dtype=jnp.float32),
            "dones": jnp.asarray(dataset['terminals'], dtype=jnp.float32),
            "masks": jnp.asarray(1.0 - dataset['terminals'], dtype=jnp.float32),
        }
        return cls(data=buffer), state_mean, state_std

    @property
    def size(self):
        return self.data["observations"].shape[0]

    @functools.partial(jax.jit, static_argnames='batch_size')
    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key=key, shape=(batch_size, ), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.dataset_id,
        save_code=False,
    )
    rng = jax.random.PRNGKey(config.seed)
    new_rng, buffer_key = jax.random.split(rng, 2)
    
    eval_env = gym.make(config.dataset_id)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    replay_buffer, state_mean, state_std = ReplayBuffer.create_from_d4rl(eval_env)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    
    value_def = ensemblize(MLP, 2)(hidden_dims=(256, 256))
    sample_batch = replay_buffer.sample_batch(key=buffer_key, batch_size=1)
    iql_learner = create_learner(seed=0, observations=sample_batch['observations'], 
                                 actions=sample_batch['actions'], value_def=value_def)
    
    for step in tqdm(range(1, config.max_timesteps + 1)):
        new_key, buffer_key = jax.random.split(buffer_key, 2)
        batch = replay_buffer.sample_batch(key=buffer_key, batch_size=config.batch_size)
        iql_learner, statistics = iql_learner.update(batch)
        wandb.log(statistics)
        
        if step % config.eval_freq == 0 or step == config.max_timesteps - 1:
            eval_returns, norm_returns = evaluate_d4rl(eval_env, iql_learner, config.n_episodes, seed=config.seed)
            wandb.log({"Normalized D4RL return": norm_returns.mean()})
            wandb.log({"Returns": eval_returns.mean()})

def eval_actor(agent, obs):
    return agent.actor(obs, ).mean()

def evaluate_d4rl(env: gym.Env, actor: IQLAgent, num_episodes: int, seed: int):
    env.seed(seed)
    print("Evaluating Agent")
    
    returns = []
    normalized_returns = []
    for _ in tqdm(range(num_episodes)):
        obs, done = env.reset(), False
        total_reward = 0.0
        
        while not done:
            action = eval_actor(actor, obs)
            obs, reward, done, _ = env.step(jax.device_get(action))
            print(action)
            total_reward += reward
            
        returns.append(total_reward)
        normalized_returns.append(env.get_normalized_score(total_reward) * 100.0)
    return np.array(returns), np.asarray(normalized_returns)

if __name__ == "__main__":
    train()
    
def get_default_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 10.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    return config