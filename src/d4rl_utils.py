import d4rl
import gym
import numpy as np

from typing import *
from jaxrl_m.dataset import Dataset

def valid_goal_sampler(self, np_random):
    valid_cells = []

    for i in range(len(self._maze_map)):
      for j in range(len(self._maze_map[0])):
        if self._maze_map[i][j] in [0, 'r', 'g']:
          valid_cells.append((i, j))

    sample_choices = valid_cells
    cell = sample_choices[np_random.choice(len(sample_choices))]
    xy = self._rowcol_to_xy(cell, add_random_noise=True)

    random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

    xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

    return xy

def compute_mean_std(states: np.ndarray, eps: float):
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize(states, mean, std):
    return (states - mean) / std

def get_dataset(env: gym.Env,
                gcrl: bool = False,
                dataset=None,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                filter_terminals=False,
                obs_dtype=np.float32,
                normalize_states=False,
                normalize_rewards=False
                ):
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dataset['terminals'][-1] = 1
        if filter_terminals:
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

        if 'antmaze' in env.spec.id.lower() and gcrl:
            dones_float = np.zeros_like(dataset['rewards'])
            dataset['terminals'][:] = 0.

            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
        else:
            dones_float = dataset['terminals'].copy()

        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        if normalize_states:
            state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
            observations = normalize(dataset["observations"], state_mean, state_std)
            next_observations = normalize(dataset["next_observations"], state_mean, state_std)
        
        if normalize_rewards:
            dataset = modify_reward(dataset, env_name=env.spec.id)
            print(f"Rewards mean: {dataset['rewards'].mean()}")
            
        return Dataset.create(
            observations=observations,
            next_observations=next_observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
        )

def modify_reward(
    dataset: Dict[str, np.ndarray], env_name: str, max_episode_steps: int = 1000
):
    if any(s in env_name.lower() for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = get_normalization(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return dataset

def get_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset['rewards'], dataset['dones_float']):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000
