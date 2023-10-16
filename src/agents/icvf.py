import jax
import jax.numpy as jnp

import equinox as eqx
import equinox.nn as nn
import optax

import dataclasses

from jaxrl_m.eqx_common import TargetTrainState
from typing import * 
import ml_collections

@jax.jit
def expectile_loss(adv, diff, expectile):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2

def icvf_loss(value_fn, target_model_fn, agent, batch, expectile: float = 0.9, discount: float = 0.999):
    (next_v1_gz, next_v2_gz) = agent.evaluate_ensemble(target_model_fn, batch['next_observations'], batch['icvf_goals'], batch['icvf_desired_goals']) # s, g, z (z = s+)

    q1_gz = batch['icvf_rewards'] + discount * batch['icvf_masks'] * next_v1_gz
    q2_gz = batch['icvf_rewards'] + discount * batch['icvf_masks'] * next_v2_gz
    q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)

    (v1_gz, v2_gz) = agent.evaluate_ensemble(value_fn, batch['observations'], batch['icvf_goals'], batch['icvf_desired_goals'])

    (next_v1_zz, next_v2_zz) = agent.evaluate_ensemble(target_model_fn, batch['next_observations'], batch['icvf_goals'], batch['icvf_desired_goals'])
    next_v_zz = (next_v1_zz + next_v2_zz) / 2
    
    q_zz = batch['icvf_desired_rewards'] + discount * batch['icvf_desired_masks'] * next_v_zz

    (v1_zz, v2_zz) = agent.evaluate_ensemble(target_model_fn, batch['observations'], batch['icvf_desired_goals'], batch['icvf_desired_goals'])
    v_zz = (v1_zz + v2_zz) / 2
    adv = q_zz - v_zz
    
    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, expectile).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, expectile).mean()
    value_loss = value_loss1 + value_loss2

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v_gz max': v1_gz.max(),
        'v_gz min': v1_gz.min(),
        'v_zz': v_zz.mean(),
        'v_gz': v1_gz.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
        'reward mean': batch['icvf_rewards'].mean(),
        'q_gz max': q1_gz.max()
    }


class ICVF_Multilinear(eqx.Module):
    phi_net: eqx.Module
    psi_net: eqx.Module
    T_net: eqx.Module
    
    matrix_a: eqx.Module # same as in ICVF paper, use low-rank approx?
    matrix_b: eqx.Module
    
    psi_ln: eqx.Module
    phi_ln: eqx.Module
    
    def __init__(self, key, in_size, hidden_dims, use_layer_norm: bool = False):
        phi_key, psi_key, t_key, matrix_a_key, matrix_b_key = jax.random.split(key, 5)
        
        self.phi_net = nn.MLP(in_size=in_size, out_size=hidden_dims[-1],
                              depth=3, width_size=hidden_dims[0], key=phi_key, activation=jax.nn.gelu)
        if use_layer_norm:
            self.phi_ln = nn.LayerNorm(hidden_dims[-1])
            self.psi_ln = nn.LayerNorm(hidden_dims[-1])
        else:
            self.phi_ln = nn.Identity()
            self.psi_ln = nn.Identity()

        self.psi_net = nn.MLP(in_size=in_size, out_size=hidden_dims[-1],
                              depth=3, width_size=hidden_dims[0], key=psi_key, activation=jax.nn.gelu)
        
        self.T_net = nn.MLP(in_size=hidden_dims[-1], out_size=hidden_dims[-1],
                              depth=3, width_size=hidden_dims[0], key=t_key, activation=jax.nn.gelu)
        
        self.matrix_a = nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_a_key)
        self.matrix_b = nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_b_key)
        
    def __call__(self, s, s_plus, intent):
        phi = self.phi_ln(self.phi_net(s))
        psi = self.psi_ln(self.psi_net(s_plus))
        z = self.psi_net(intent) # z = psi(s_z), s_z here like in paper subset of state space (s_z - some state)
        T_z = self.T_net(z)
        
        phi_z = self.matrix_a(T_z * phi)
        psi_z = self.matrix_b(T_z * psi)
        v = (phi_z * psi_z).sum(axis=-1)
        
        return v

class ICVF_Agent(eqx.Module):
    value: eqx.Module
    loss_fn: Callable
    
    @staticmethod
    @eqx.filter_vmap(in_axes=dict(model=eqx.if_array(0), s=None, s_plus=None, intent=None))
    @eqx.filter_vmap(in_axes=dict(model=None, s=0, s_plus=0, intent=0))
    def evaluate_ensemble(model, s, s_plus, intent):
        return model(s, s_plus, intent)
    
    @eqx.filter_jit
    def pretrain_update(self, batch, seed=None):
        (val, aux_data), grads = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)(self.value.model, self.value.target_model, self, batch)
        new_value = self.value.apply_updates(grads).soft_update(tau=0.005)
        return dataclasses.replace(self, value=new_value), aux_data
    
def create_learner(
    seed,
    observations,
    hidden_dims: Sequence[int] = (256, 256, 256),
    optim_kwargs: dict = {
                    'learning_rate': 3e-3, # 1e-3 for FC, for vision 5e-4
                    'eps': 0.0003125
                 },
    num_ensemble_vals: int = 2,
    use_layer_norm: bool = False
):
    rng = jax.random.PRNGKey(seed)
    rng, critic_key = jax.random.split(rng, 2)
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return ICVF_Multilinear(key=keys, in_size=observations.shape[-1],
                                hidden_dims=hidden_dims, use_layer_norm=use_layer_norm)
    
    icvf_ensemble = ensemblize(jax.random.split(critic_key, num_ensemble_vals))
    icvf_target_ensemble = ensemblize(jax.random.split(critic_key, num_ensemble_vals))
    
    value = TargetTrainState.create(model=icvf_ensemble, target_model=icvf_target_ensemble,
                                           optim=optax.adam(**optim_kwargs))
    
    return ICVF_Agent(value=value, loss_fn=icvf_loss)
    

def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 0.001,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.85,
    })

    return config