import ml_collections
import equinox as eqx
import equinox.nn as nn
import distrax
import optax
import functools
import copy
import dataclasses

from jaxrl_m.eqx_common import TrainState, TargetTrainState
import jax
import jax.numpy as jnp
from jaxrl_m.typing import *
from jaxtyping import PyTree

class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())
    
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
    
class ValueFunction(eqx.Module):
    mlp: nn.MLP
    
    def __init__(self, key, state_dim, hidden_dim, hidden_depth):
        self.mlp = nn.MLP(in_size=state_dim, out_size=1,
                          width_size=hidden_dim,
                          depth=hidden_depth,
                          key=key)
    @jax.jit
    def __call__(self, x):
        return self.mlp(x)

class Actor(eqx.Module):
    mlp: nn.MLP
    max_action: float
    
    def __init__(self, key, state_dim, hidden_dim, hidden_depth, act_dim):
        self.mlp = nn.MLP(in_size=state_dim, out_size=act_dim * 2,
                          width_size=hidden_dim,
                          depth=hidden_depth,
                          final_activation=jax.nn.leaky_relu,
                          key=key)
    @eqx.filter_jit
    def __call__(self, x):
        mu, log_sigma = jnp.split(self.mlp(x), 2, axis=-1)
        log_sigma = jnp.clip(log_sigma, -5, 2)
        dist = FixedDistrax(TanhNormal, mu, jnp.exp(log_sigma))
        return dist
    
    @eqx.filter_jit
    def act(self, obs):
        dist = self(obs)
        return dist.mean()

class SimpleEncoder(eqx.Module):
    def __init__(self, key, state_dim, hidden_dim, hidden_depth, rep_dim):
        self.mlp = nn.MLP(in_size=state_dim, out_size=rep_dim,
                          width_size=hidden_dim,
                          depth=hidden_depth,
                          key=key)
    @eqx.filter_jit
    def __call__(self, x):
        return self.mlp(x)

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
                   actions: jnp.ndarray,
                   lr: float = 3e-4,
                   use_rep: int = 1,
                   temperature: float = 1,
                   actor_hidden_dims: Sequence[int] = (256, 256),
                   value_hidden_dims: Sequence[int] = (256, 256),
                   discount: float = 0.99,
                   tau: float = 0.005,
                   discrete: int = 0,
                   pretrain_expectile: float = 0.85,
                   use_layer_norm: int = 0,
                   rep_type: str = 'state',
                   rep_dim: int = 128,
                   encoder: str = "impala", # for images only
                   visual: int = 0, # for images only
                   **kwargs):
    
    print(f'Extra kwargs: {kwargs}')
            
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, value_key, encoder_key = jax.random.split(rng, 4)
    
    encoder = None
    if use_rep:
        encoder = SimpleEncoder(key=encoder_key, rep_dim=rep_dim,
                                hidden_dim=encoder_hidden_dims[0], hidden_depth=len(encoder_hidden_dims))
    state_dim = rep_dim if use_rep else observations.shape[-1]
    
    @eqx.filter_vmap
    def ensemblize_value_fn(rngs):
        return ValueFunction(key=rngs,state_dim=state_dim,
                                hidden_dim=value_hidden_dims[0], hidden_depth=len(value_hidden_dims))
    
    value_fn = ensemblize_value_fn(jax.random.split(value_key, 2))
    actor_fn = Actor(key=actor_key, state_dim=state_dim, hidden_dim=actor_hidden_dims[0],
                     hidden_depth=len(actor_hidden_dims), act_dim=actions.shape[-1])
    
    network_def = CilotAgent(
        encoders={
            'base_encoder': encoder
        },
        networks={
            'value': value_fn,
            'target_value': copy.deepcopy(value_fn),
            'actor': actor_fn
        }
    )
    network_tx = optax.adam(learning_rate=lr)
    network = TrainState.create(network_def, optim=network_tx)
    return TrainerCilotAgent(network=network)



def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.85,
    })

    return config