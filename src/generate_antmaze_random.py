import numpy as np
import h5py
import argparse
from d4rl.locomotion import maze_env, ant, swimmer
from d4rl.locomotion.wrappers import NormalizedBoxEnv
from PIL import Image
import os
from tqdm.auto import tqdm


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    data_dict['next_observations'] = data_dict['observations'][1:].copy()
    return data_dict

def reset_data():
    return {'observations': [],
            'next_observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, next_s, a, r, tgt, done, timeout, env_data):
    data['observations'].append(s)
    data['next_observations'].append(next_s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict

def combine_ds(num_expert_trajs, num_agent_trajs, expert_ds, agent_ds):
    #dict_keys(['observations', 'actions', 'next_observations', 'rewards', 'terminals'])
    # maybe use other approach as in https://github.com/JasonMa2016/SMODICE/blob/main/utils.py#L110
    mixed_ds = {
        'observations': np.concatenate((agent_ds['observations'][:num_agent_trajs], expert_ds['observations'][:num_expert_trajs]), dtype=np.float32),
        'next_observations': np.concatenate((agent_ds['next_observations'][:num_agent_trajs], expert_ds['next_observations'][:num_expert_trajs]), dtype=np.float32),
        'rewards': np.concatenate((agent_ds['rewards'][:num_agent_trajs], expert_ds['rewards'][:num_expert_trajs]), dtype=np.float32),
        'actions': np.concatenate((agent_ds['actions'][:num_agent_trajs], expert_ds['actions'][:num_expert_trajs]), dtype=np.float32),
        'terminals': np.concatenate((agent_ds['terminals'][:num_agent_trajs], expert_ds['terminals'][:num_expert_trajs]), dtype=np.float32),
    }
    return mixed_ds


def obtain_agent_ds(
    noisy: bool = True,
    maze: str = "umaze",
    num_samples: int = int(300_000),
    env: str = "ant",
    max_episode_steps: int = 1_000,
    policy_file: str = "",
    multi_start: bool = False,
    multigoal: bool = False
):
    if maze == 'umaze':
        maze = maze_env.U_MAZE
    elif maze == 'medium':
        maze = maze_env.BIG_MAZE
    elif maze == 'large':
        maze = maze_env.HARDEST_MAZE
    elif maze == 'umaze_eval':
        maze = maze_env.U_MAZE_EVAL
    elif maze == 'medium_eval':
        maze = maze_env.BIG_MAZE_EVAL
    elif maze == 'large_eval':
        maze = maze_env.HARDEST_MAZE_EVAL
    else:
        raise NotImplementedError
    
    env = NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze, reward_type="sparse", maze_size_scaling=4.0, non_zero_reset=multi_start))
    
    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    
    ts = 0
    num_episodes = 0
    for _ in range(num_samples):
        act = env.action_space.sample()

        if noisy:
            act = act + np.random.randn(*act.shape)*0.2
            act = np.clip(act, -1.0, 1.0)

        ns, r, done, info = env.step(act)
        timeout = False
        if ts >= max_episode_steps:
            timeout = True
            #done = True
        append_data(data, s[:-2], ns[:-2], act, r, env.target_goal, done, timeout, env.physics.data)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1

        if done or timeout:
            done = False
            ts = 0
            s = env.reset()
            env.set_target_goal()
            num_episodes += 1
        else:
            s = ns
    
    fname = 'antmaze-umaze-v2-randomstart-noiserandomaction.hdf5'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')
    return get_dataset(fname)

if __name__ == '__main__':
    obtain_agent_ds()
