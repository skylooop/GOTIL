import jax.numpy as jnp
import ml_collections
from jaxrl_m.eqx_common import *
from jaxrl_m.eqx_networks import *
from jaxrl_m.typing import *




class SACAgent:
    def train(self,):
        pass


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99, # = gamma
        tau: float = 5e-3,
        temperature: float = 1,
        discrete: int = 0,
        use_layer_norm: int = 0,
        **kwargs):
    print('Extra kwargs:', kwargs)
    key = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    
    actor = TrainState.create(
        model=Actor(observations.shape[-1], actions.shape[-1], actor_hidden_dims, key=actor_key),
        optim=optax.adam(learning_rate=lr)
    )
    return SACAgent()
