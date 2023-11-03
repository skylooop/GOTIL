from jaxrl_m.typing import *

import jax
from jaxrl_m.common import TrainStateEQX
import equinox as eqx
from src.agents.icvf import update
import dataclasses

class JointGotilAgent(eqx.Module):
    expert_icvf: TrainStateEQX
    agent_icvf: TrainStateEQX
    config: dict
    
    def pretrain_expert(self, pretrain_batch):
        agent, update_info =  update(self.expert_icvf, pretrain_batch)
        return dataclasses.replace(self, expert_icvf=agent), update_info
    
    def pretrain_agent(self, pretrain_batch):
        agent, update_info =  update(self.agent_icvf, pretrain_batch)
        return dataclasses.replace(self, agent_icvf=agent), update_info

def create_eqx_learner(seed: int,
                       expert_icvf,
                       agent_icvf,
                        discount: float = 0.95,
                        target_update_rate: float = 0.005,
                        expectile: float = 0.9,
                        no_intent: bool = False,
                        min_q: bool = True,
                        periodic_target_update: bool = False,
                        **kwargs):
    rng = jax.random.PRNGKey(seed)
    
    config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
    return JointGotilAgent(expert_icvf=expert_icvf, agent_icvf=agent_icvf, config=config)
    
    