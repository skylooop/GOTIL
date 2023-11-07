# OS params
import os
import hydra
import warnings
import pickle
import rootutils
import functools
import gzip

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
warnings.filterwarnings("ignore")

# Configs & Printing
import wandb
from omegaconf import DictConfig
ROOT = rootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"],
                            pythonpath=True, cwd=True)

# Libs
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tqdm.auto import tqdm
from jaxrl_m.wandb import setup_wandb

from utils.rich_utils import print_config_tree
from src.agents import hiql, icvf, gotil
from src.agents.gotil import evaluate_with_trajectories_gotil
from src.agents.icvf import update, eval_ensemble
from src.gc_dataset import GCSDataset
from src.utils import record_video
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from xmagical_ext import xmagical_utils

from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor
from xmagical_ext.xmagical_utils import evaluate_with_trajectories_xmagical

@jax.jit
def get_debug_statistics_hiql(agent, batch):
    def get_info(s, g):
        return agent.network(s, g, info=True, method='value')

    s = batch['observations']
    g = batch['goals']
    info = get_info(s, g)
    stats = {}
    stats.update({
        'v': info['v'].mean(),
    })

    return stats

@jax.jit
def get_gcvalue(agent, s, g):
    v1, v2 = agent.network(s, g, method='value')
    return (v1 + v2) / 2

def get_v(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal)

@eqx.filter_jit
def get_debug_statistics_icvf(agent, batch):
    def get_info(s, g, z):
        if agent.config['no_intent']:
            return agent.value(s, g, jnp.ones_like(z), method='get_info')
        else:
            return eval_ensemble(agent.value_learner.model, s, g, z)
    s = batch['observations']
    g = batch['icvf_goals']
    z = batch['icvf_desired_goals']

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz.mean(),
        'v_szz': info_szz.mean(),
        'v_sgz': info_sgz.mean(),
        'v_sgg': info_sgg.mean(),
        'v_szg': info_szg.mean(),
        'diff_szz_szg': (info_szz - info_szg).mean(),
        'diff_sgg_sgz': (info_sgg - info_sgz).mean(),
        # 'v_ssz': info_ssz['v'].mean(),
        # 'v_szz': info_szz['v'].mean(),
        # 'v_sgz': info_sgz['v'].mean(),
        # 'v_sgg': info_sgg['v'].mean(),
        # 'v_szg': info_szg['v'].mean(),
        # 'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean(),
        # 'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean(),
    })
    return stats

@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(s[None], g[None], method='value')
        return (v1 + v2) / 2
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

@hydra.main(version_base="1.4", config_path=str(ROOT/"configs"), config_name="base.yaml")
def main(config: DictConfig):
    print_config_tree(config)
    setup_wandb(hyperparam_dict=dict(config),
                      project=config.logger.project,
                      group=config.logger.group,
                      name=None)
    goal_info = None
    discrete = False
    if 'antmaze' in config.Env.dataset_id.lower():
        env_name = config.Env.dataset_id.lower()
        if 'ultra' in env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
        else:
            import d4rl
            import gym
            env = gym.make(env_name)
        env = EpisodeMonitor(env)
        # RUN generate_antmaze_random.py
        # offline ds - random noisy data + some successful trajs
        # offline_dataset = get_dataset("d4rl_ext/antmaze_demos/antmaze-umaze-v2-randomstart-noiserandomaction.hdf5")
        # offline_dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset=offline_dataset)
        # TODO: Add to ds above some successful trajs
        
        dataset = d4rl_utils.get_dataset(env, 
                                         gcrl=config.Env.gcrl, # for GCRL, just change terminals
                                         normalize_states=config.Env.normalize_states,
                                         normalize_rewards=config.Env.normalize_rewards)
        
        os.environ['CUDA_VISIBLE_DEVICES']="4" # for headless server
        env.render(mode='rgb_array', width=200, height=200)
        os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4"
        
        if 'large' in env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
            
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=config.Env.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)

        elif 'ultra' in env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

    elif "gripper" in env_name:
        env = xmagical_utils.make_env(config.Env.modality, visual="State")
        if config.Env.video_type == 'same':
            dataset = xmagical_utils.get_dataset(config.Env.modality, dir_name=config.Env.xmagical_expert_path)
        elif config.Env.video_type == 'cross':
            dataset = xmagical_utils.crossembodiment_dataset(config.Env.modality, config.Env.xmagical_expert_path)
        else:
            dataset = xmagical_utils.crossembodiment_dataset(None, config.Env.xmagical_expert_path)

    elif 'kitchen' in env_name:
        env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    
    elif 'calvin' in env_name:
        from src.envs.calvin import CalvinEnv
        from hydra import compose, initialize
        from src.envs.gym_env import GymWrapper
        from src.envs.gym_env import wrap_env

        initialize(config_path='src/envs/conf')
        cfg = compose(config_name='calvin')
        env = CalvinEnv(**cfg)
        env.max_episode_steps = cfg.max_episode_steps = 360
        env = GymWrapper(
            env=env,
            from_pixels=cfg.pixel_ob,
            from_state=cfg.state_ob,
            height=cfg.screen_size[0],
            width=cfg.screen_size[1],
            channels_first=False,
            frame_skip=cfg.action_repeat,
            return_state=False,
        )
        env = wrap_env(env, cfg)

        data = pickle.load(gzip.open('data/calvin.gz', "rb"))
        ds = []
        for i, d in enumerate(data):
            if len(d['obs']) < len(d['dones']):
                continue  # Skip incomplete trajectories.
            # Only use the first 21 states of non-floating objects.
            d['obs'] = d['obs'][:, :21]
            new_d = dict(
                observations=d['obs'][:-1],
                next_observations=d['obs'][1:],
                actions=d['actions'][:-1],
            )
            num_steps = new_d['observations'].shape[0]
            new_d['rewards'] = np.zeros(num_steps)
            new_d['terminals'] = np.zeros(num_steps, dtype=bool)
            new_d['terminals'][-1] = True
            ds.append(new_d)
        dataset = dict()
        for key in ds[0].keys():
            dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
        dataset = d4rl_utils.get_dataset(None, env_name, dataset=dataset)
    elif 'procgen' in env_name:
        from src.envs.procgen_env import ProcgenWrappedEnv, get_procgen_dataset
        import matplotlib

        matplotlib.use('Agg')

        n_processes = 1
        env_name = 'maze'
        env = ProcgenWrappedEnv(n_processes, env_name, 1, 1)

        if env_name == 'procgen-500':
            dataset = get_procgen_dataset('data/procgen/level500.npz', state_based=('state' in env_name))
            min_level, max_level = 0, 499
        elif env_name == 'procgen-1000':
            dataset = get_procgen_dataset('data/procgen/level1000.npz', state_based=('state' in env_name))
            min_level, max_level = 0, 999
        else:
            raise NotImplementedError

        # Test on large levels having >=20 border states
        large_levels = [12, 34, 35, 55, 96, 109, 129, 140, 143, 163, 176, 204, 234, 338, 344, 369, 370, 374, 410, 430, 468, 470, 476, 491] + \
            [5034, 5046, 5052, 5080, 5082, 5142, 5244, 5245, 5268, 5272, 5283, 5335, 5342, 5366, 5375, 5413, 5430, 5474, 5491]
        goal_infos = []
        goal_infos.append({'eval_level': [level for level in large_levels if min_level <= level <= max_level], 'eval_level_name': 'train'})
        goal_infos.append({'eval_level': [level for level in large_levels if level > max_level], 'eval_level_name': 'test'})

        dones_float = 1.0 - dataset['masks']
        dones_float[-1] = 1.0
        dataset = dataset.copy({
            'dones_float': dones_float
        })

        discrete = True
        example_action = np.max(dataset['actions'], keepdims=True)
    else:
        raise NotImplementedError

    env.reset()
    if config.GoalDS:
        gc_dataset = GCSDataset(dataset, **dict(config.GoalDS),
                                discount=config.Env.discount)
        if config.algo.algo_name == "icvf" or config.algo.algo_name == "gotil":
            from src.icvf_utils import DebugPlotGenerator
            visualizer = DebugPlotGenerator(env_name, gc_dataset)
        if 'antmaze' in env_name:
            example_trajectory = gc_dataset.sample(50, indx=np.arange(1000, 1050), mode=config.algo.algo_name)
        elif 'kitchen' in env_name:
            example_trajectory = gc_dataset.sample(50, indx=np.arange(0, 50), mode=config.algo.algo_name)
        elif 'calvin' in env_name:
            example_trajectory = gc_dataset.sample(50, indx=np.arange(0, 50), mode=config.algo.algo_name)
        elif 'procgen-500' in env_name:
            example_trajectory = gc_dataset.sample(50, indx=np.arange(5000, 5050), mode=config.algo.algo_name)
        elif 'procgen-1000' in env_name:
            example_trajectory = gc_dataset.sample(50, indx=np.arange(5000, 5050), mode=config.algo.algo_name)
        else:
            pass
            #TODO: Add new environments
            
    total_steps = config.pretrain_steps 
    example_batch = dataset.sample(1)

    if config.algo.algo_name == "hiql":
        agent = hiql.create_learner(seed=config.seed,
                                    observations=example_batch['observations'],
                                    actions=example_batch['actions'] if not discrete else example_action,
                                    discrete=discrete,
                                    **dict(config.algo))
        
    elif config.algo.algo_name == "icvf":
        agent = icvf.create_eqx_learner(config.seed,
                                       example_batch['observations'],
                                       discount=config.Env.discount,
                                       **dict(config.algo))
        
    elif config.algo.algo_name == "gotil":
        
        from src.generate_antmaze_random import obtain_agent_ds, get_dataset, combine_ds
        if config.algo.path_to_agent_data == "":
            agent_dataset = obtain_agent_ds()
            print("Dataset obtained")
            exit()
        else:
            print("Loading Agent dummy dataset")
            agent_dataset = get_dataset(config.algo.path_to_agent_data)
            expert_dataset = d4rl.qlearning_dataset(env)
            mixed_ds = combine_ds(config.algo.num_expert_trajs, config.algo.num_agent_trajs, expert_dataset, agent_dataset)
            #here rewards mean are 0, change smth to get non zero?
            mixed_ds = d4rl_utils.get_dataset(env, dataset=mixed_ds,
                                         gcrl=config.Env.gcrl,
                                         normalize_states=config.Env.normalize_states,
                                         normalize_rewards=config.Env.normalize_rewards)
            agent_gc_dataset = GCSDataset(mixed_ds, **dict(config.GoalDS), discount=config.Env.discount)
            
        expert_icvf = icvf.create_eqx_learner(config.seed,
                                       observations=example_batch['observations'],
                                       discount=config.Env.discount,
                                       load_pretrained_phi=True,
                                       **dict(config.algo))
        agent_icvf = icvf.create_eqx_learner(config.seed,
                                       observations=example_batch['observations'], # TODO: Replace with random actions from SMODICE
                                       discount=config.Env.discount,
                                       **dict(config.algo))
        
        agent = gotil.create_eqx_learner(config.seed,
                                        expert_icvf=expert_icvf,
                                        agent_icvf=agent_icvf,
                                        batch_size=config.batch_size,
                                        observations=example_batch['observations'],
                                        actions=example_batch['actions'],
                                        **dict(config.algo))
        expert_training_icvf = True
    
    rng = jax.random.PRNGKey(config.seed)
    for i in tqdm(range(1, total_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Training"):
        pretrain_batch = gc_dataset.sample(config.batch_size, mode=config.algo.algo_name)
            
        if config.algo.algo_name == "gotil": 
            # if i < total_steps / 2:
            #     agent, update_info = agent.pretrain_expert(pretrain_batch)
            # else:
            # save expert
            expert_training_icvf = False
            agent_dataset_batch = agent_gc_dataset.sample(config.batch_size, mode=config.algo.algo_name)
            agent, update_info, rng = agent.pretrain_agent(agent_dataset_batch, rng)
                
        elif config.algo.algo_name == "hiql":
            agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch)
        else:
            agent, update_info = update(agent, pretrain_batch)
            
        if i % config.log_interval == 0 and config.algo.algo_name == "hiql":
            debug_statistics = get_debug_statistics_hiql(agent, pretrain_batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'training_stats/{k}': v for k, v in debug_statistics.items()})
            wandb.log(train_metrics, step=i)
            
        if i % config.log_interval == 0 and (config.algo.algo_name == "icvf" or config.algo.algo_name == "gotil"):
            if config.algo.algo_name == "gotil":
                if expert_training_icvf:
                    debug_statistics = get_debug_statistics_icvf(agent.expert_icvf, pretrain_batch)
                else:
                    debug_statistics = get_debug_statistics_icvf(agent.agent_icvf, pretrain_batch)
            else:
                debug_statistics = get_debug_statistics_icvf(agent, pretrain_batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            wandb.log(train_metrics, step=i)
            
        if i % config.save_interval == 0 and config.algo.algo_name == "icvf":
            unensemble_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, agent.value_learner.model)
            with open("icvf_model.eqx", "wb") as f:
                eqx.tree_serialise_leaves(f, unensemble_model.phi_net)
                
        if i % config.eval_interval == 0 and (config.algo.algo_name == "icvf" or config.algo.algo_name == "gotil"):
            if config.algo.algo_name == "gotil":
                if expert_training_icvf:
                    visualizations =  visualizer.generate_debug_plots(agent.expert_icvf)
                else:
                    visualizations =  visualizer.generate_debug_plots(agent.agent_icvf)
                     
            else:
                visualizations = visualizer.generate_debug_plots(agent)
            eval_metrics = {f'visualizations/{k}': v for k, v in visualizations.items()}
            wandb.log(eval_metrics, step=i)
            
            if config.algo.algo_name == "gotil":
                base_observation = jax.tree_map(lambda arr: arr[0], gc_dataset.dataset['observations'])
                evaluate_with_trajectories_gotil()
                
        if i % config.eval_interval == 0 and config.algo.algo_name == "hiql":
            policy_fn = functools.partial(supply_rng(agent.sample_actions), discrete=discrete)
            high_policy_fn = functools.partial(supply_rng(agent.sample_high_actions))
            policy_rep_fn = agent.get_policy_rep
            base_observation = jax.tree_map(lambda arr: arr[0], gc_dataset.dataset['observations'])
            
            if 'procgen' in env_name:
                eval_metrics = {}
                for goal_info in goal_infos:
                    eval_info, trajs, renders = evaluate_with_trajectories(
                        policy_fn, high_policy_fn, policy_rep_fn, env, env_name=env_name, num_episodes=config.eval_episodes,
                        base_observation=base_observation, num_video_episodes=0,
                        use_waypoints=config.algo.use_waypoints,
                        eval_temperature=0, epsilon=0.05,
                        goal_info=goal_info, config=dict(config.algo),
                    )
                    eval_metrics.update({f'evaluation/level{goal_info["eval_level_name"]}_{k}': v for k, v in eval_info.items()})
            
            elif 'gripper' in env_name:
                indx = jax.numpy.argmax(gc_dataset.dataset['masks'] == 0)
                goal = jax.tree_map(lambda arr: arr[indx], gc_dataset.dataset['observations'])
                eval_info, video, goal = evaluate_with_trajectories_xmagical(
                    policy_fn, high_policy_fn, policy_rep_fn, env, num_episodes=config.eval_episodes,
                    base_observation=base_observation, goal=goal, num_video_episodes=config.num_video_episodes,
                    use_waypoints=config.algo.use_waypoints,
                    eval_temperature=0,
                    goal_info=goal_info, config=dict(config.algo))
                
            else:
                eval_info, trajs, renders = evaluate_with_trajectories(
                    policy_fn, high_policy_fn, policy_rep_fn, env, env_name=env_name, num_episodes=config.eval_episodes,
                    base_observation=base_observation, num_video_episodes=config.num_video_episodes,
                    use_waypoints=config.algo.use_waypoints,
                    eval_temperature=0,
                    goal_info=goal_info, config=dict(config.algo),
                )
                eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}

                if config.num_video_episodes > 0:
                    video = record_video('Video', i, renders=renders)
                    eval_metrics['video'] = video

            traj_metrics = get_traj_v(agent, example_trajectory)
            value_viz = viz_utils.make_visual_no_image(
                traj_metrics,
                [functools.partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            )
            eval_metrics['value_traj_viz'] = wandb.Image(value_viz)

            if 'antmaze' in env_name and 'large' in env_name and 'antmaze' in env_name:
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)
                
                image_v = d4rl_ant.gcvalue_image(
                    viz_env,
                    viz_dataset,
                    functools.partial(get_v, agent),
                )
                eval_metrics['v'] = wandb.Image(image_v)

            wandb.log(eval_metrics, step=i)

if __name__ == '__main__':
    main()
