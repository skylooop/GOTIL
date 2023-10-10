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

class JointICVF(eqx.Module):
    expert_icvf: eqx.Module
    agent_icvf: eqx.Module
    cur_processor: str = None

    def pretrain_expert(self, batch):
        new_expert_agent, update_info = supply_rng(self.expert_icvf.pretrain_update)(batch)
        return dataclasses.replace(self, expert_icvf=new_expert_agent, cur_processor="expert"), update_info
    
    def pretrain_agent(self, batch):
        new_agent, update_info = supply_rng(self.agent_icvf.pretrain_update)(batch)
        return dataclasses.replace(self, agent_icvf=new_agent, cur_processor="agent"), update_info

def create_joint_learner(seed: int,
                   offline_ds_obs,
                   expert_ds_obs,
                   encoder,
                   intention_book,
                   actor_hidden_dims: Sequence[int] = (256, 256),
                   value_hidden_dims: Sequence[int] = (256, 256),
                   discount: float = 0.99,
                   tau: float = 0.005,
                   pretrain_expectile: float = 0.85,
                   **kwargs):
    
    rng = jax.random.PRNGKey(seed)
    expert_icvf = create_learner(seed, expert_ds_obs)
    agent_icvf = create_learner(seed, offline_ds_obs)

    return JointICVF(expert_icvf=expert_icvf, agent_icvf=agent_icvf)

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