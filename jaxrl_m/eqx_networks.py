import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from jaxtyping import PyTree
from typing import Dict, Any, Tuple
import distrax

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

class Actor(eqx.Module):
    layers: nn.Sequential

    def __init__(self, obs_dim, action_dim, hidden_dim, *, key):
        keys = jax.random.split(key, num=4)
        self.layers = nn.Sequential([
            nn.Linear(obs_dim, hidden_dim, key=keys[0]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[2]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, action_dim * 2, key=keys[3])
        ])

    def __call__(self, obs):
        mu, log_sigma = jnp.split(self.layers(obs), 2, axis=-1)
        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = jnp.clip(log_sigma, -5, 2)
        dist = FixedDistrax(TanhNormal, mu, jnp.exp(log_sigma))
        return dist