import h5py
from tqdm import tqdm
import d4rl
import gym

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

    return data_dict

env = gym.make("antmaze-umaze-v2")
offline_ds = get_dataset("/home/simmax21/Desktop/GOTIL/d4rl_ext/antmaze_demos/antmaze-umaze-v2-randomstart-noiserandomaction.hdf5")
#expert_ds = d4rl.qlearning_dataset(env)
print(offline_ds)
#print(expert_ds['rewards'].sum())