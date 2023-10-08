import h5py
from tqdm import tqdm
import d4rl
import gym


env = gym.make("antmaze-umaze-v2")
ds = d4rl.qlearning_dataset(env)
print(ds)