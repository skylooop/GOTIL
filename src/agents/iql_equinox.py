import dataclasses
from dataclasses import dataclass, asdict
from typing import *
import contextlib
import functools

import warnings
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp
import distrax

from tqdm.auto import tqdm
import numpy as np
import equinox as eqx
import optax
from jaxtyping import *

import gym
import d4rl

import wandb
import pyrallis

@dataclass
class TrainConfig:
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL_EQX"
    dataset_id: str = "antmaze-large-diverse-v2"
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 10.0
    iql_tau: float = 0.9 #experctile
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
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
    seed: int = 42

class TrainState(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    optim_state: optax.OptState

    @classmethod
    def create(cls, *, model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state,
                   **kwargs)
    
    @eqx.filter_jit
    def apply_updates(self, grads):
        updates, new_optim_state = self.optim.update(grads, self.optim_state)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )

class TrainTargetState(TrainState):
    target_model: eqx.Module

    @classmethod
    def create(cls, *, model, target_model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state, target_model=target_model,
                   **kwargs)

    def soft_update(self, tau: float = 0.005):
        model_params = eqx.filter(self.model, eqx.is_array)
        target_model_params, target_model_static = eqx.partition(self.target_model, eqx.is_array)

        new_target_params = optax.incremental_update(model_params, target_model_params, tau)
        return dataclasses.replace(
            self,
            model=self.model,
            target_model=eqx.combine(new_target_params, target_model_static)
        )
    
    def apply_updates(self, grads):
        updates, new_optim_state = self.optim.update(grads, self.optim_state)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )

class FixedDistrax(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def sample_and_log_prob(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample_and_log_prob(seed=seed)

    def sample(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample(seed=seed)

    def log_prob(self, x):
        return self.cls(*self.args, **self.kwargs).log_prob(x)

    def mean(self):
        return self.cls(*self.args, **self.kwargs).mean()
    
    
class ReplayBuffer(eqx.Module):
    data: Dict[str, jax.Array]

    @classmethod
    def create_from_d4rl(cls, dataset) -> "ReplayBuffer":
        buffer = {
            "observations": jnp.asarray(dataset["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(dataset["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(dataset["rewards"], dtype=jnp.float32),
            "next_observations": jnp.asarray(dataset["next_observations"], dtype=jnp.float32),
            "dones": jnp.asarray(dataset['terminals'], dtype=jnp.float32),
        }
        return cls(data=buffer)

    @property
    def size(self):
        return self.data["observations"].shape[0]

    @functools.partial(jax.jit, static_argnames=["batch_size"])
    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    
    def normalize_state(state):
        # if 'antmaze' in env.spec.id.lower():
        #     return (state['observation'] - state_mean) / state_std
        # else:
        #     
        return (state - state_mean) / state_std

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# def qlearning_dataset(dataset: minari.MinariDataset) -> Dict[str, np.ndarray]:
#     obs, next_obs, actions, rewards, dones = [], [], [], [], []

#     for idx, episode in enumerate(dataset):
#         obs.append(episode.observations['observation'][:-1].astype(jnp.float32)) # fix for other than antmaze
#         next_obs.append(episode.observations['observation'][1:].astype(jnp.float32))
#         actions.append(episode.actions.astype(jnp.float32))
#         episode.rewards[:-1] = False
#         rewards.append(episode.rewards)
#         dones.append(episode.terminations)
#     return {
#         "observations": jnp.concatenate(obs),
#         "actions": jnp.concatenate(actions),
#         "next_observations": jnp.concatenate(next_obs),
#         "rewards": jnp.concatenate(rewards),
#         "terminals": jnp.concatenate(dones),
#     }

def return_reward_range(
    dataset: Dict[str, np.ndarray], max_episode_steps: int
) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(
    dataset: Dict[str, np.ndarray], env_name: str, max_episode_steps: int = 1000
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

class QNet(eqx.Module):
    hidden_dims: tuple[int] = (256, 256)
    net: eqx.Module
    
    def __init__(self, key, state_dim, action_dim):
        key, mlp_key = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(in_size=state_dim + action_dim, 
                              out_size=1, depth=len(self.hidden_dims), width_size=self.hidden_dims[-1],
                              key=mlp_key)
    
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        return self.net(x)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), state=None, action=None))
@eqx.filter_vmap(in_axes=dict(ensemble=None, state=0, action=0))
def eval_ensemble(ensemble, state, action):
    return ensemble(state, action)

class VNet(eqx.Module):
    hidden_dims: tuple[int] = (256, 256)
    net: eqx.Module
    
    def __init__(self, key, state_dim):
        key, mlp_key = jax.random.split(key, 2)
        self.net = eqx.nn.MLP(in_size=state_dim, 
                              out_size=1, depth=len(self.hidden_dims), width_size=self.hidden_dims[-1],
                              key=mlp_key)
    
    def __call__(self, obs):
        return self.net(obs)

class GaussianPolicy(eqx.Module):
    net: eqx.Module
    log_std_min: int = -5.0
    log_std_max: int = 2.0
    temperature: float = 10.0
    log_stds: Array

    def __init__(self, key, state_dim, action_dim, hidden_dims):
        key_mean, key_log_std = jax.random.split(key, 2)
        
        self.net = eqx.nn.MLP(in_size=state_dim,
                              out_size=action_dim,
                              width_size=hidden_dims[0],
                              depth=len(hidden_dims),
                              key=key_mean)
        
        self.log_stds = jax.numpy.zeros(shape=(action_dim, ))
    
    def __call__(self, state):
        means = self.net(state)
        log_sigma = jnp.clip(self.log_stds, self.log_std_min, self.log_std_max)
        dist = FixedDistrax(distrax.MultivariateNormalDiag, means, jnp.exp(log_sigma) * self.temperature)
        return dist
      
def expectile_loss(diff, expectile=0.9):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IQLagent(eqx.Module):
    q_learner: TrainState
    v_learner: TrainTargetState
    actor_learner: TrainState
    
    @eqx.filter_jit
    def eval_actor(self, obs):
        return self.actor_learner.model(obs).mean()
    
@eqx.filter_jit
def update_agent(agent, batch):
    def v_loss_fn(v_net):
        q1, q2 = eval_ensemble(agent.q_learner.model, batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = eqx.filter_vmap(v_net)(batch['observations'])
        value_loss = expectile_loss(q - v).mean()
        return value_loss, {
            'value_loss': value_loss,
        }
    
    def q_loss_fn(q_net):
        next_v = eqx.filter_vmap(agent.v_learner.target_model)(batch['next_observations'])
        target = batch['rewards'][:, None] + 0.99 * (1.0 - batch['dones'][:, None]) * next_v
        q1, q2 = eval_ensemble(q_net, batch['observations'], batch['actions'])
        q_loss = ((q1 - target)**2 + (q2 - target)**2).mean()
        return q_loss, {
            'q_loss': q_loss,
        }
    
    def actor_loss_fn(actor_net):
        v = eqx.filter_vmap(agent.v_learner.model)(batch['observations'])
        q1, q2 = eval_ensemble(agent.q_learner.model, batch['observations'], batch['actions'])
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * 10.0)
        exp_a = jnp.minimum(exp_a, 100.0)
        dist = eqx.filter_vmap(actor_net)(batch['observations'])
        
        log_probs = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_probs).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss,
        }    
    
    (val_v, aux_v), v_grads = eqx.filter_value_and_grad(v_loss_fn, has_aux=True)(agent.v_learner.model)
    updated_v_learner = agent.v_learner.apply_updates(v_grads).soft_update()
    
    (val_actor, aux_actor), actor_grads = eqx.filter_value_and_grad(actor_loss_fn, has_aux=True)(agent.actor_learner.model)
    updated_actor_learner = agent.actor_learner.apply_updates(actor_grads)

    (val_q, aux_q), q_grads = eqx.filter_value_and_grad(q_loss_fn, has_aux=True)(agent.q_learner.model)
    updated_q_learner = agent.q_learner.apply_updates(q_grads)

    return dataclasses.replace(
        agent,
        q_learner = updated_q_learner,
        v_learner = updated_v_learner,
        actor_learner = updated_actor_learner
    ), {**aux_v, **aux_q, **aux_actor}

@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.dataset_id,
        save_code=False,
    )
    # minari.download_dataset(config.dataset_id)
    # dataset = minari.load_dataset(config.dataset_id)
    
    # eval_env = dataset.recover_environment()
    # if 'antmaze' not in config.dataset_id:
    eval_env = gym.make(config.dataset_id)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    # else:
    #     state_dim = eval_env.observation_space['observation'].shape[0]
    #     action_dim = eval_env.action_space.shape[0]

    qdataset = d4rl.qlearning_dataset(eval_env)
    if config.normalize_reward:
        modify_reward(qdataset, config.dataset_id)
    
    if config.normalize_state:
        state_mean, state_std = compute_mean_std(qdataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
        
    qdataset["observations"] = normalize_states(
        qdataset["observations"], state_mean, state_std
    )
    qdataset["next_observations"] = normalize_states(
        qdataset["next_observations"], state_mean, state_std
    )
    
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer.create_from_d4rl(qdataset)
    key = jax.random.PRNGKey(seed=config.seed)
    key, q_key, val_key, val_key_target, actor_key, buffer_key = jax.random.split(key, 6)
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return QNet(key=keys, state_dim=state_dim, action_dim=action_dim)
    
    schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.actor_lr)
    actor_tx = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(schedule_fn))

    q_learner = TrainState.create(
        model=ensemblize(jax.random.split(q_key, 2)),
        optim=optax.adam(learning_rate=config.qf_lr)
    )
    v_learner = TrainTargetState.create(
        model=VNet(key=val_key, state_dim=state_dim),
        target_model=VNet(key=val_key, state_dim=state_dim),
        optim=optax.adam(learning_rate=config.vf_lr)
    )
    actor_learner = TrainState.create(
        model=GaussianPolicy(key=actor_key, state_dim=state_dim, action_dim=action_dim, hidden_dims=(256, 256)),
        optim=actor_tx
    )
    
    iql_agent = IQLagent(
        q_learner=q_learner,
        v_learner=v_learner,
        actor_learner=actor_learner
    )
    
    for step in tqdm(range(1, config.max_timesteps + 1)):
        # maybe add jax.lax.scan -> for each epoch update NUM_UPDATES_PER_EPOCH steps
        buf_key, buffer_key = jax.random.split(buffer_key, 2)
        batch = replay_buffer.sample_batch(key=buffer_key, batch_size=config.batch_size)

        iql_agent, statistics = update_agent(iql_agent, batch)

        wandb.log(statistics, step=step)
        
        if step % config.eval_freq == 0 or step == config.max_timesteps - 1:
            eval_returns = evaluate_d4rl(eval_env, iql_agent, config.n_episodes, seed=config.seed)
            wandb.log({"evaluation return": eval_returns.mean()}, step=step)
            
            # with contextlib.suppress(ValueError):
            #     normalized_score = minari.get_normalized_score(dataset, eval_returns).mean() * 100
            #     wandb.log({"normalized score": normalized_score}, step=step)

def evaluate(env: gym.Env, actor: IQLagent, num_episodes: int, seed: int):
    print("Evaluating Agent")
    
    returns = []
    # launch agent for num_episodes times and in each perform actions till done
    for _ in tqdm(range(num_episodes)):
        done = False
        obs, info = env.reset(seed=seed+_)
        episode_reward = 0.0
        
        while not done:
            action = actor.eval_actor(obs)
            obs, reward, terminated, truncated, info = env.step(jax.device_get(action))
            done = terminated or truncated
            episode_reward += reward
        returns.append(episode_reward) # per episode
        
    return np.asarray(returns)

def evaluate_d4rl(env: gym.Env, actor: IQLagent, num_episodes: int, seed: int):
    env.seed(seed)
    print("Evaluating Agent")
    
    returns = []
    # launch agent for num_episodes times and in each perform actions till done
    for _ in tqdm(range(num_episodes)):
        obs, done = env.reset(), False
        total_reward = 0.0
        
        while not done:
            action = actor.eval_actor(obs)
            obs, reward, done, _ = env.step(jax.device_get(action))
            total_reward += reward
        returns.append(total_reward) # per episode
    return np.array(returns)

if __name__ == "__main__":
    train()

