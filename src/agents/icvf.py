from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import optax
from jaxrl_m.common import TrainState, TrainTargetStateEQX, target_update, nonpytree_field

import flax
import flax.linen as nn

import equinox as eqx
from src.special_networks import MultilinearVF_EQX
import dataclasses

def expectile_loss(adv, diff, expectile=0.8):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2

def icvf_loss(value_fn, target_value_fn, batch, config):
    if config['no_intent']:
        batch['icvf_desired_goals'] = jax.tree_map(jnp.ones_like, batch['icvf_desired_goals'])

    ###
    # Compute TD error for outcome s_+
    # 1(s == s_+) + V(s', s_+, z) - V(s, s_+, z)
    ###

    (next_v1_gz, next_v2_gz) = eval_ensemble(target_value_fn, batch['next_observations'], batch['icvf_goals'], batch['icvf_desired_goals'])
    q1_gz = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v1_gz
    q2_gz = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v2_gz
    q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)

    (v1_gz, v2_gz) = eval_ensemble(value_fn, batch['observations'], batch['icvf_goals'], batch['icvf_desired_goals'])

    ###
    # Compute the advantage of s -> s' under z
    # r(s, z) + V(s', z, z) - V(s, z, z)
    ###

    (next_v1_zz, next_v2_zz) = eval_ensemble(target_value_fn, batch['next_observations'], batch['icvf_desired_goals'], batch['icvf_desired_goals'])
    if config['min_q']:
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
    else:
        next_v_zz = (next_v1_zz + next_v2_zz) / 2
    
    q_zz = batch['icvf_desired_rewards'] + config['discount'] * batch['icvf_desired_masks'] * next_v_zz

    (v1_zz, v2_zz) = eval_ensemble(target_value_fn, batch['observations'], batch['icvf_desired_goals'], batch['icvf_desired_goals'])
    v_zz = (v1_zz + v2_zz) / 2
    adv = q_zz - v_zz

    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    def masked_mean(x, mask):
        return (x * mask).sum() / (1e-5 + mask.sum())

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
        'reward mean': batch['rewards'].mean(),
        'mask mean': batch['masks'].mean(),
        'q_gz max': q1_gz.max(),
        'value_loss1': masked_mean((q1_gz-v1_gz)**2, batch['masks']), # Loss on s \neq s_+
        'value_loss2': masked_mean((q1_gz-v1_gz)**2, 1.0 - batch['masks']), # Loss on s = s_+
    }

def periodic_target_update(
    model: TrainState, target_model: TrainState, period: int
) -> TrainState:
    new_target_params = jax.tree_map(
        lambda p, tp: optax.periodic_update(p, tp, model.step, period),
        model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)

class ICVFAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    value: TrainState
    target_value: TrainState
    config: dict = nonpytree_field()
        
    @jax.jit
    def update(agent, batch):
        def value_loss_fn(value_params):
            value_fn = lambda s, g, z: agent.value(s, g, z, params=value_params)
            target_value_fn = lambda s, g, z: agent.target_value(s, g, z)

            return icvf_loss(value_fn, target_value_fn, batch, agent.config)
        
        if agent.config['periodic_target_update']:
            new_target_value = periodic_target_update(agent.value, agent.target_value, int(1.0 / agent.config['target_update_rate']))
        else:
            new_target_value = target_update(agent.value, agent.target_value, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        return agent.replace(value=new_value, target_value=new_target_value), value_info
    
def create_learner(
                 seed: int,
                 observations: jnp.ndarray,
                 value_def: nn.Module,
                 optim_kwargs: dict = {
                    'learning_rate': 0.00005,
                    'eps': 0.0003125
                 },
                 discount: float = 0.95,
                 target_update_rate: float = 0.005,
                 expectile: float = 0.9,
                 no_intent: bool = False,
                 min_q: bool = True,
                 periodic_target_update: bool = False,
                **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        
        value_params =  value_def.init(rng, observations, observations, observations).pop('params')
        value = TrainState.create(value_def, value_params, tx=optax.adam(**optim_kwargs))
        target_value = TrainState.create(value_def, value_params)

        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        ))

        return ICVFAgent(rng=rng, value=value, target_value=target_value, config=config)

class ICVF_EQX_Agent(eqx.Module):
    value_learner: TrainTargetStateEQX
    config: dict

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None), out_axes=0)
def eval_ensemble(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble)(s, g, z)

@eqx.filter_jit
def update(agent, batch):
    (val, value_aux), v_grads = eqx.filter_value_and_grad(icvf_loss, has_aux=True)(agent.value_learner.model, agent.value_learner.target_model, batch, agent.config)
    updated_v_learner = agent.value_learner.apply_updates(v_grads).soft_update()
    return dataclasses.replace(agent, value_learner=updated_v_learner, config=agent.config), value_aux
    
def create_eqx_learner(seed: int,
                       observations: jnp.array,
                       hidden_dims: list,
                       optim_kwargs: dict = {
                            'learning_rate': 0.00005,
                            'eps': 0.0003125
                        },
                        discount: float = 0.95,
                        target_update_rate: float = 0.005,
                        expectile: float = 0.9,
                        no_intent: bool = False,
                        min_q: bool = True,
                        periodic_target_update: bool = False,
                        **kwargs):
        print('Extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        
        @eqx.filter_vmap
        def ensemblize(keys):
            return MultilinearVF_EQX(key=keys, state_dim=observations.shape[-1], hidden_dims=hidden_dims)
        
        value_learner = TrainTargetStateEQX.create(
            model=ensemblize(jax.random.split(rng, 2)),
            target_model=ensemblize(jax.random.split(rng, 2)),
            optim=optax.adam(**optim_kwargs)
        )
        config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
        return ICVF_EQX_Agent(value_learner=value_learner, config=config)
