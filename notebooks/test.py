import jax
import equinox as eqx
import equinox.nn as nn
import functools
import math
from jaxrl_m.typing import *

rng = jax.random.PRNGKey(25)
rng, random_key = jax.random.split(rng, 2)
x = jax.random.normal(random_key, shape=(5, 2))

@eqx.filter_vmap
def make_ensemble(key):
    return nn.MLP(2,1,256,4, key=key, activation=jax.nn.selu)

mlp_ensemble = make_ensemble(jax.random.split(rng, 2))

@eqx.filter_vmap(in_axes=dict(model=eqx.if_array(0), x=None), out_axes=0)
@eqx.filter_vmap(in_axes=dict(model=None, x=0))
def evaluate_ensemble(model, x):
    return model(x)

print(x)

print(evaluate_ensemble(mlp_ensemble, x))

class Critic(eqx.Module):
    layers: nn.Sequential

    def __init__(self, key, hidden_dim=256):
        keys = jax.random.split(key, num=4)
        self.layers = nn.Sequential([
            nn.Linear(2, hidden_dim, key=keys[0], use_bias=False),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[1], use_bias=False),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[2],use_bias=False),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, 1, key=keys[3],use_bias=False)
        ])

    def __call__(self, obs):
        out = self.layers(obs)
        return out

#net = nn.MLP(2,1,128,3, key=rng)
net = Critic(rng)
print(eqx.filter_vmap(net)(x))