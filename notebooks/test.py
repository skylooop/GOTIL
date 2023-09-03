# import gym
# import d4rl
import warnings
warnings.filterwarnings('ignore')

# env = gym.make("antmaze-large-diverse-v2")
# env.reset()
# dataset = d4rl.qlearning_dataset(env)
# for i in range(dataset['observations'].shape[0]):
#     env.step(dataset['actions'][i])
#     print(dataset['rewards'][i])
#     print(dataset['terminals'][i])
#     env.render()

import gymnasium as gym
import minari
env = gym.make("PointMaze_UMazeDense-v3", render_mode="human")
dataset = minari.load_dataset("pointmaze-umaze-dense-v1")
episode = dataset.sample_episodes(n_episodes=1)[0]
print(episode)
env.reset(seed=42)
for i in range(episode.actions.shape[0]):
    env.step(episode.actions[i])
    print(episode.rewards[i])
    print(episode.terminations[i])
    print(episode.truncations[i])
    env.render()

