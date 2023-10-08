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

class CilotAgent(eqx.Module):
    encoders: Dict[str, eqx.Module]
    networks: Dict[str, eqx.Module]
    
    def value(self, obs, goals):
        if self.encoders['base_encoder'] is not None:
            obs = eqx.filter_vmap(self.encoders['base_encoder'], in_axes=(eqx.if_array(0), None))(obs)
            goals = eqx.filter_vmap(self.encoders['base_encoder'], in_axes=(eqx.if_array(0), None))(goals)
        return eqx.filter_vmap(self.networks['value'])(obs, goals)
    
    def target_value(self, obs, goals):
        if self.encoders['base_encoder'] is not None:
            obs = eqx.filter_vmap(self.encoders['base_encoder'], in_axes=(eqx.if_array(0), None))(obs)
            goals = eqx.filter_vmap(self.encoders['base_encoder'], in_axes=(eqx.if_array(0), None))(goals)
        return eqx.filter_vmap(self.networks['target_value'])(obs, goals)
    
    def actor(self, obs):
        if self.encoders['base_encoder'] is not None:
            obs = eqx.filter_vmap(self.encoders['base_encoder'], in_axes=(eqx.if_array(0), None))(obs)
        return eqx.filter_vmap(self.networks['actor'])(obs)
    
    def __call__(self, obs, goals):
        return {
            'value': self.value(obs, goals),
            'target_value': self.value(obs, goals),
            'actor': self.actor(obs)
        }

class TrainerCilotAgent(eqx.Module):
    network: eqx.Module
    target_update_rate: float = 5e-3
        
    def pretrain_update(self, pretrain_batch, seed=None):
        def loss_fn():
            info = {}
            
        self.network.model.networks['target_value'] = jax.tree_map(
            lambda m, tp: m * self.target_update_rate + tp * (1 - self.target_update_rate), self.network.model.networks['value'], self.network.model.networks['target_value']
        )
        
        
        return dataclasses.replace(self,
                                   network=new_network)
    pretrain_update = jax.jit(pretrain_update, static_argnums=(0, ))
    
def create_learner(seed: int,
                   observations: jnp.ndarray,
                   use_rep: int = False, # for future VQVAE
                   actor_hidden_dims: Sequence[int] = (256, 256),
                   value_hidden_dims: Sequence[int] = (256, 256),
                   discount: float = 0.99,
                   tau: float = 0.005,
                   pretrain_expectile: float = 0.85,
                   **kwargs):
    
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, value_key, encoder_key = jax.random.split(rng, 4)
    
    if not use_rep:
        encoder = SimpleEncoder(key=encoder_key, rep_dim=rep_dim,
                                hidden_dim=encoder_hidden_dims[0], hidden_depth=len(encoder_hidden_dims))


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