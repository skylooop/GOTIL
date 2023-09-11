import ml_collections
import equinox as eqx
import equinox.nn as nn
import distrax
import optax

from jaxrl_m.eqx_common import TrainState, CriticTrainState
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
    
    def __init__(self, key, state_dim, hidden_dim, hidden_depth, act_dim, max_action):
        self.mlp = nn.MLP(in_size=state_dim, out_size=act_dim * 2,
                          width_size=hidden_dim,
                          depth=hidden_depth,
                          final_activation=jax.nn.relu,
                          key=key)
    @jax.jit
    def __call__(self, x):
        mu, log_sigma = jnp.split(self.mlp(x), 2, axis=-1)
        log_sigma = jnp.clip(log_sigma, -5, 2)
        dist = FixedDistrax(TanhNormal, mu, jnp.exp(log_sigma))
        return dist
    
    @jax.jit
    def act(self, obs):
        dist = self(obs)
        return dist.mean()
    
class CilotAgent(eqx.Module):
    value_fn: eqx.Module
    actor_fn: eqx.Module
    
    def __init__(self, value_fn, actor_fn):
        pass
    
    @jax.jit
    def pretrain_update(self, pretrain_batch):
        def loss_fn():
            #actor_update
            pass


def create_learner(seed: int,
                   observations: jnp.ndarray,
                   actions: jnp.ndarray,
                   max_action: float,
                   hidden_dim: int,
                   hidden_depth: int,
                   actor_lr: float = 3e-4,
                   value_lr: float = 3e-4,
                   use_rep: int = 0,
                   **kwargs):
    
    print(f'Extra kwargs: {kwargs}')
    
    @eqx.filter_vmap
    def ensemblize(rngs):
        return ValueFunction(key=rngs,state_dim=observations.shape[-1],
                                hidden_dim=hidden_dim, hidden_depth=hidden_depth)
        
    rng = jax.random.PRNGKey(seed)  
    rng, actor_key, value_key = jax.random.split(rng, 3)
    ## make encoder for states
    actor_fn = Actor(key=actor_key, state_dim=observations.shape[-1], hidden_dim=hidden_dim,
                     hidden_depth=hidden_depth, act_dim=actions.shape[-1], max_action=max_action)
    
    value = CriticTrainState.create(model=ensemblize(jax.random.split(value_key, 2)), 
                                    target_model=ensemblize(jax.random.split(value_key, 2)),
                                    optim=optax.adam(value_lr))
    actor = TrainState.create(model=actor_fn, optim=optax.adam(actor_lr))
    
    
    return CilotAgent() ## make encoder for states