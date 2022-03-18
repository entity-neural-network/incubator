from typing import Dict, Iterator, Mapping, Optional, Protocol, Tuple
import numpy.typing as npt
import numpy as np
import torch
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool
from entity_gym.simple_trace import Tracer
from entity_gym.environment import VecActionMask


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
