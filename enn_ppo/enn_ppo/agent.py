from typing import Dict, Iterator, Mapping, Optional, Protocol, Tuple

import numpy as np
import numpy.typing as npt
import torch
from ragged_buffer import RaggedBufferBool, RaggedBufferF32, RaggedBufferI64

import entity_gym.agent
from entity_gym.environment import Action, Observation, VecActionMask
from entity_gym.environment.env_list import action_index_to_actions
from entity_gym.environment.environment import ActionType
from entity_gym.environment.vec_env import batch_obs
from entity_gym.simple_trace import Tracer
from rogue_net.rogue_net import RogueNet


class PPOAgent(Protocol):
    def get_action_and_auxiliary(
        self,
        entities: Mapping[str, RaggedBufferF32],
        visible: Mapping[str, RaggedBufferBool],
        action_masks: Mapping[str, VecActionMask],
        tracer: Tracer,
        prev_actions: Optional[Dict[str, RaggedBufferI64]] = None,
    ) -> Tuple[
        Dict[str, RaggedBufferI64],  # actions
        Dict[str, torch.Tensor],  # action probabilities
        Dict[str, torch.Tensor],  # entropy
        Dict[str, npt.NDArray[np.int64]],  # number of actors in each frame
        Dict[str, torch.Tensor],  # auxiliary head values
        Dict[str, torch.Tensor],  # full logits
    ]:
        ...

    def to(self, device: torch.device) -> "PPOAgent":
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        ...

    def get_auxiliary_head(
        self,
        entities: Mapping[str, RaggedBufferF32],
        visible: Mapping[str, RaggedBufferBool],
        head_name: str,
        tracer: Tracer,
    ) -> torch.Tensor:
        ...


class RogueNetAgent(entity_gym.agent.Agent):
    def __init__(self, agent: RogueNet):
        self.agent = agent

    def act(self, obs: Observation) -> Tuple[Dict[ActionType, Action], float]:
        vec_obs = batch_obs([obs], self.agent.obs_space, self.agent.action_space)
        with torch.no_grad():
            act_indices, _, _, _, aux, logits = self.agent.get_action_and_auxiliary(
                vec_obs.features,
                vec_obs.visible,
                vec_obs.action_masks,
                tracer=Tracer(False),
            )
        actions = action_index_to_actions(
            self.agent.obs_space,
            self.agent.action_space,
            act_indices,
            obs,
            probs={k: l.exp().cpu().numpy() for k, l in logits.items()},
        )
        return actions, float(aux["value"].item())
