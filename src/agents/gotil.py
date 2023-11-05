from jaxrl_m.typing import *

import jax
from jaxrl_m.common import TrainTargetStateEQX, TrainStateEQX
from src.agents.iql_equinox import GaussianPolicy
import equinox as eqx
from src.agents.icvf import update, eval_ensemble
import dataclasses
import optax
from src.special_networks import MonolithicVF_EQX
from tqdm.auto import tqdm

# Optimal Transport Imports
import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import plot, sinkhorn_divergence
from tqdm.auto import tqdm
from ott.utils import tqdm_progress_fn

class JointGotilAgent(eqx.Module):
    expert_icvf: TrainTargetStateEQX
    agent_icvf: TrainTargetStateEQX
    value_net: TrainTargetStateEQX
    actor_intents_learner: TrainStateEQX
    agent_intent_cb: Array
    config: dict
    
    @eqx.filter_jit
    def sample_intentions():
        pass
    
    @eqx.filter_jit
    def sample_actions():
        pass
    
    def pretrain_expert(self, pretrain_batch):
        agent, update_info = update(self.expert_icvf, pretrain_batch)
        return dataclasses.replace(self, expert_icvf=agent), update_info
    
    def pretrain_agent(self, pretrain_batch):
        agent, agent_info = update(self.agent_icvf, pretrain_batch)
        #value_net, value_info = gotil_value_update(self.value_net, pretrain_batch, self.config) # V(s, z)
        agent_updated_codebook, agent_updated_v, ot_info = update_ot(self.expert_icvf, self.value_net, pretrain_batch, self.agent_intent_cb)
        updated_actor_intents_learner, actor_info = update_actor(self.actor_intents_learner, pretrain_batch, agent_updated_codebook, agent_updated_v, self.expert_icvf)
        return dataclasses.replace(self, agent_icvf=agent, value_net=agent_updated_v, agent_intent_cb=agent_updated_codebook, actor_intents_learner=updated_actor_intents_learner), ot_info

@eqx.filter_jit
def update_actor(actor_intents_learner, batch, agent_codebook, agent_value, expert_icvf):
    def intention_actor_loss(actor_intents_learner):
        # target - expert intentions
        v1_a, v2_a = eval_value_ensemble(agent_value.model, batch['observations'], agent_codebook)
        nv1, nv2 = eval_value_ensemble(agent_value.model, batch['next_observations'], agent_codebook)
        v = (v1_a + v2_a) / 2
        nv = (nv1 + nv2) / 2
        
        adv = nv - v
        exp_a = jnp.exp(adv * 10.0)
        exp_a = jnp.minimum(exp_a, 100.0)
        dist = eqx.filter_vmap(actor_intents_learner)(batch['observations'])
        target = jnp.minimum(eval_ensemble(expert_icvf.value_learner.model, batch['observations'], agent_codebook, agent_codebook))
        log_probs = dist.log_prob(target)
        actor_intents_loss = -(exp_a * log_probs).mean()
        
        return actor_intents_loss, {
            'high_actor_loss': actor_intents_loss,
            'high_adv': adv.mean(),
    }

    def actor_loss():
        pass
    
    (val_intents_actor, aux_intents), actor_intents_grads = eqx.filter_value_and_grad(intention_actor_loss, has_aux=True)(actor_intents_learner.model)
    updated_actor_intents = actor_intents_learner.apply_updates(actor_intents_grads)
    
    return updated_actor_intents
    
def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), z=None))
def get_expert_intents(ensemble, z):
    return eqx.filter_vmap(ensemble)(z)

@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), obs=None, z=None))
def eval_value_ensemble(ensemble, obs, z):
    return eqx.filter_vmap(ensemble)(obs, z)

def sink_div(combined, b, batch):
    geom, agent_value = combined
    agent_marginals, _ = eval_value_ensemble(agent_value, batch['observations'], batch['icvf_desired_goals'])
    agent_marginals = jax.nn.softmax(agent_marginals, axis=0).squeeze()
    ot = sinkhorn_divergence.sinkhorn_divergence(
        geom,
        x=geom.x,
        a=agent_marginals,
        b=b,
        y=geom.y,
        static_b=True,
        sinkhorn_kwargs={"use_danskin": True},
    )
    return ot.divergence, ot

def ot_update(x, y, agent_value, b, cost_fn, batch, num_iter: int = 100, lr: float = 0.2, epsilon:float = 0.25):
    cost_fn_vg = eqx.filter_jit(eqx.filter_value_and_grad(cost_fn, has_aux=True))
    
    for i in tqdm(range(0, num_iter + 1), desc="Computing OT"):
        geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
        (cost, ot), (geom_g, geom_value_grads) = cost_fn_vg((geom, agent_value.model), b, batch)
        assert ot.converged
        x = x - geom_g.x * lr
        agent_value = agent_value.apply_updates(geom_value_grads).soft_update()
    return x, agent_value, ot.divergence

def update_ot(expert_icvf, agent_value, batch, agent_intent_cb):
    expert_intents, _ = get_expert_intents(expert_icvf.value_learner.model.psi_net, batch['icvf_desired_goals']) # change z here
    # agent_marginals, _ = eval_value_ensemble(agent_value.model, batch['observations'], batch['icvf_desired_goals'])
    # agent_marginals = jax.nn.softmax(agent_marginals, axis=0).squeeze()
    
    expert_marginals, _ = eval_ensemble(expert_icvf.value_learner.model, batch['observations'], batch['icvf_desired_goals'], batch['icvf_desired_goals'])
    expert_marginals = jax.nn.softmax(expert_marginals, axis=0).squeeze()
    
    agent_updated_codebook, agent_updated_marginals, ot_info = ot_update(x=agent_intent_cb, y=expert_intents, agent_value=agent_value, b=expert_marginals, cost_fn=sink_div, batch=batch)
    return agent_updated_codebook, agent_updated_marginals, ot_info
    
    
def create_eqx_learner(seed: int,
                       expert_icvf,
                       agent_icvf,
                       observations,
                       batch_size: int,
                       hidden_dims: list = [256, 256],
                       intent_codebook_dim: int = 256,
                       discount: float = 0.99,
                       target_update_rate: float = 0.005,
                       expectile: float = 0.9,
                       no_intent: bool = False,
                       min_q: bool = True,
                       periodic_target_update: bool = False,
                       **kwargs):
    
    rng = jax.random.PRNGKey(seed)
    rng, value_model, actor_learner_key = jax.random.split(rng, 3)
    agent_intent_cb = jax.random.normal(key=rng, shape=(batch_size, intent_codebook_dim), dtype=jnp.float32) # init intentions for dummy agent
    
    @eqx.filter_vmap
    def ensemblize(keys):
        return MonolithicVF_EQX(key=keys, state_dim=observations.shape[-1], hidden_dims=hidden_dims)
    
    value_net_def = TrainTargetStateEQX.create(model=ensemblize(jax.random.split(value_model, 2)),
                                                                target_model=ensemblize(jax.random.split(value_model, 2)),
                                                                optim=optax.adam(learning_rate=3e-4))
    actor_intents_learner = TrainStateEQX.create(
        model=GaussianPolicy(key=actor_learner_key,
                             hidden_dims=[256, 256],
                             state_dim=observations.shape[-1],
                             action_dim=intent_codebook_dim),
        optim=optax.adam(learning_rate=3e-4)
    )
    config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
    return JointGotilAgent(expert_icvf=expert_icvf, agent_icvf=agent_icvf, actor_intents_learner=actor_intents_learner,
                           value_net=value_net_def, config=config,
                           agent_intent_cb=agent_intent_cb)
    


# def value_loss(value_net, value_target, batch, config):
#     (next_v1, next_v2) = eval_value_ensemble(value_target, batch['next_observations'], batch['icvf_desired_goals'])
#     next_v = jnp.minimum(next_v1, next_v2)
#     q = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v
    
#     (v1_t, v2_t) = eval_value_ensemble(value_target, batch['observations'], batch['icvf_desired_goals'])
#     v_t = (v1_t + v2_t) / 2
#     adv = q - v_t
    
#     q1 = batch['rewards'] + config['discount'] * batch['icvf_masks'] * next_v1
#     q2 = batch['rewards'] + config['discount'] * batch['icvf_masks'] * next_v2
#     (v1, v2) = eval_value_ensemble(value_net, batch['observations'], batch['icvf_desired_goals'])
#     value_loss1 = expectile_loss(adv, q1 - v1, config['expectile']).mean()
#     value_loss2 = expectile_loss(adv, q2 - v2, config['expectile']).mean()
#     value_loss = value_loss1 + value_loss2
#     advantage = adv
    
#     return value_loss, {
#         'Gotil_value_loss': value_loss,
#         'Gotil_abs adv mean': jnp.abs(advantage).mean(),
#         'Gotil_adv mean': advantage.mean(),
#         'Gotil_adv max': advantage.max(),
#         'Gotil_adv min': advantage.min(),
#         'Gotil_accept prob': (advantage >= 0).mean(),
#     }

# @eqx.filter_jit
# def gotil_value_update(value_net, batch, config):
#     (val, value_aux), v_grads = eqx.filter_value_and_grad(value_loss, has_aux=True)(value_net.model, value_net.target_model, batch, config)
#     updated_v_learner = value_net.apply_updates(v_grads)
#     updated_v_target = updated_v_learner.soft_update()
#     return dataclasses.replace(value_net, model=updated_v_learner, target_model=updated_v_target), value_aux