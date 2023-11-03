from jaxrl_m.typing import *

import jax
from jaxrl_m.common import TrainTargetStateEQX
import equinox as eqx
from src.agents.icvf import update
import dataclasses
import optax
from src.special_networks import MonolithicVF_EQX


class JointGotilAgent(eqx.Module):
    expert_icvf: TrainTargetStateEQX
    agent_icvf: TrainTargetStateEQX
    value_net: TrainTargetStateEQX
    agent_intent_cb: Array
    config: dict
    
    def pretrain_expert(self, pretrain_batch):
        agent, update_info = update(self.expert_icvf, pretrain_batch)
        return dataclasses.replace(self, expert_icvf=agent), update_info
    
    def pretrain_agent(self, pretrain_batch):
        agent, agent_info = update(self.agent_icvf, pretrain_batch)
        value_net, value_info = gotil_value_update(self.value_net, pretrain_batch, self.config) # V(s, z)
        updated_cb, book_info = update_ot(self.expert_icvf, agent, value_net, pretrain_batch, self.agent_intent_cb)
        return dataclasses.replace(self, agent_icvf=agent, value_net=value_net, agent_intent_cb=updated_cb), value_info

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def value_loss(value_net, value_target, batch, config):
    (next_v1, next_v2) = eval_value_ensemble(value_target, batch['next_observations'], batch['icvf_desired_goals'])
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v
    
    (v1_t, v2_t) = eval_value_ensemble(value_target, batch['observations'], batch['icvf_desired_goals'])
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t
    
    q1 = batch['rewards'] + config['discount'] * batch['icvf_masks'] * next_v1
    q2 = batch['rewards'] + config['discount'] * batch['icvf_masks'] * next_v2
    (v1, v2) = eval_value_ensemble(value_net, batch['observations'], batch['icvf_desired_goals'])
    value_loss1 = expectile_loss(adv, q1 - v1, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2
    advantage = adv
    
    return value_loss, {
        'Gotil_value_loss': value_loss,
        'Gotil_abs adv mean': jnp.abs(advantage).mean(),
        'Gotil_adv mean': advantage.mean(),
        'Gotil_adv max': advantage.max(),
        'Gotil_adv min': advantage.min(),
        'Gotil_accept prob': (advantage >= 0).mean(),
    }
    
@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), z=None))
def get_expert_intents(ensemble, z):
    return eqx.filter_vmap(ensemble)(z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), obs=None, z=None))
def eval_value_ensemble(ensemble, obs, z):
    return eqx.filter_vmap(ensemble)(obs, z)

@eqx.filter_jit
def gotil_value_update(value_net, batch, config):
    (val, value_aux), v_grads = eqx.filter_value_and_grad(value_loss, has_aux=True)(value_net.model, value_net.target_model, batch, config)
    updated_v_learner = value_net.apply_updates(v_grads)
    updated_v_target = updated_v_learner.soft_update()
    return dataclasses.replace(value_net, model=updated_v_learner, target_model=updated_v_target), value_aux

@eqx.filter_jit
def update_ot(expert_icvf, agent_icvf, batch, agent_intent_cb):
    pass
    # #def ot_update()
    
    # # obtain intents from expert (just psi_theta(desired_outcome))
    # expert_intents, _ = get_expert_intents(expert_icvf.value_learner.model.psi_net, batch['icvf_desired_goals'])
    # #ot_update(expert_intents, agent_intent_cb)
    # #return 
    
    
def create_eqx_learner(seed: int,
                       expert_icvf,
                       agent_icvf,
                       observations,
                       batch_size: int,
                       hidden_dims: list = [256, 256],
                       intent_codebook_size: int = 256,
                       discount: float = 0.99,
                       target_update_rate: float = 0.005,
                       expectile: float = 0.9,
                       no_intent: bool = False,
                       min_q: bool = True,
                       periodic_target_update: bool = False,
                       **kwargs):
    
    rng = jax.random.PRNGKey(seed)
    rng, value_model = jax.random.split(rng, 2)
    agent_intent_cb = jax.random.normal(key=rng, shape=(batch_size, intent_codebook_size), dtype=jnp.float32)
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return MonolithicVF_EQX(key=keys, state_dim=observations.shape[-1], hidden_dims=hidden_dims)
    
    value_net_def = TrainTargetStateEQX.create(model=ensemblize(jax.random.split(value_model, 2)),
                                                                target_model=ensemblize(jax.random.split(value_model, 2)),
                                                                optim=optax.adam(learning_rate=3e-4))
    config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
    return JointGotilAgent(expert_icvf=expert_icvf, agent_icvf=agent_icvf, value_net=value_net_def, config=config, agent_intent_cb=agent_intent_cb)
    
    