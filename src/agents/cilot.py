import ml_collections
import equinox as eqx
import equinox.nn as nn
import optax
import functools
import copy
import dataclasses

import jax
import jax.numpy as jnp
from jaxrl_m.typing import *
from jaxtyping import PyTree
from jaxrl_m.evaluation import supply_rng

from src.agents.icvf import create_learner

class Critic(eqx.Module):
    def __init__(self, obs):
        self.net = nn.MLP(in_size=obs * 2, out_size=1, width_size=256, depth=2) # (256, 256)
        
    def __call__(self, observations, intents):
        v1, v2 = self.net(jnp.concatenate([observations, intents], axis=-1)).squeeze(-1)
        return (v1 + v2) / 2.

class JointICVF(eqx.Module):
    expert_icvf: eqx.Module
    agent_icvf: eqx.Module
    agent_critic: eqx.Module
    
    cur_processor: str = None
    
    def pretrain_expert(self, batch):
        new_expert_agent, update_info = supply_rng(self.expert_icvf.pretrain_update)(batch)
        return dataclasses.replace(self, expert_icvf=new_expert_agent, cur_processor="expert"), update_info
    
    def pretrain_agent(self, batch):
        new_agent, update_info = supply_rng(self.agent_icvf.pretrain_update)(batch)
        return dataclasses.replace(self, agent_icvf=new_agent, cur_processor="agent"), update_info
    
    def expert_codebook(self):
        pass

def create_joint_learner(seed: int,
                   offline_ds_obs,
                   expert_ds_obs,
                   encoder,
                   intention_book,
                   actor_hidden_dims: Sequence[int] = (256, 256),
                   value_hidden_dims: Sequence[int] = (256, 256),
                   **kwargs):
    
    rng = jax.random.PRNGKey(seed)
    expert_icvf_key, agent_icvf_key = jax.random.split(rng, 2)
    
    expert_icvf = create_learner(expert_icvf_key, expert_ds_obs)
    agent_icvf = create_learner(agent_icvf_key, offline_ds_obs)
    agent_critic = Critic(offline_ds_obs.shape[-1])
    
    return JointICVF(expert_icvf=expert_icvf, agent_icvf=agent_icvf, agent_critic=agent_critic)

def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'tau': 0.005,
        'pretrain_expectile': 0.85,
    })

    return config