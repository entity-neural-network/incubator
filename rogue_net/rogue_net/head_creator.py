from typing import Dict, Optional, Tuple
from ragged_buffer import RaggedBufferI64
from rogue_net.ragged_tensor import RaggedTensor
from torch import nn
from torch.distributions.categorical import Categorical
import torch
import numpy as np
import numpy.typing as npt
from entity_gym.environment import (
    ActionMaskBatch,
    ActionSpace,
    CategoricalActionMaskBatch,
    CategoricalActionSpace,
    DenseSelectEntityActionMask,
    SelectEntityActionSpace,
)
from typing import Dict


class ActionHead(nn.Module):
    def forward(
        self,
        x: RaggedTensor,
        index_offsets: RaggedBufferI64,
        mask: ActionMaskBatch,
        prev_actions: Optional[RaggedBufferI64],
    ) -> Tuple[RaggedBufferI64, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class CategoricalActionHead(nn.Module):
    def __init__(self, d_model: int, n_choice: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_choice = n_choice
        self.proj = layer_init(nn.Linear(d_model, n_choice), std=0.01)

    def forward(
        self,
        x: RaggedTensor,
        index_offsets: RaggedBufferI64,
        mask: ActionMaskBatch,
        prev_actions: Optional[RaggedBufferI64],
    ) -> Tuple[torch.Tensor, npt.NDArray[np.int64], torch.Tensor, torch.Tensor]:
        assert isinstance(
            mask, CategoricalActionMaskBatch
        ), f"Expected CategoricalActionMaskBatch, got {type(mask)}"
        lengths = mask.actors.size1()
        actors = torch.tensor((mask.actors + index_offsets).as_array()).to(
            x.data.device
        )
        actor_embeds = x.data[x.index_map[actors]]
        logits = self.proj(actor_embeds)
        dist = Categorical(logits=logits)
        if prev_actions is None:
            action = dist.sample()
        else:
            action = torch.tensor(prev_actions.as_array()).to(x.data.device)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, lengths, logprob, entropy


class PaddedSelectEntityActionHead(nn.Module):
    def __init__(self, d_model: int, d_qk: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.query_proj = nn.Linear(d_model, d_qk)
        self.key_proj = nn.Linear(d_model, d_qk)

    def forward(
        self,
        x: RaggedTensor,
        index_offsets: RaggedBufferI64,
        mask: ActionMaskBatch,
        prev_actions: Optional[RaggedBufferI64],
    ) -> Tuple[torch.Tensor, npt.NDArray[np.int64], torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)  # type: ignore
    return layer


def create_head_for(space: ActionSpace, d_model: int, d_qk: int) -> nn.Module:
    if isinstance(space, CategoricalActionSpace):
        return CategoricalActionHead(d_model, len(space.choices))
    elif isinstance(space, SelectEntityActionSpace):
        return PaddedSelectEntityActionHead(d_model, d_qk)
    raise NotImplementedError()


def create_value_head(d_model: int) -> nn.Module:
    value_head = nn.Linear(d_model, 1)
    value_head.weight.data.fill_(0.0)
    value_head.bias.data.fill_(0.0)
    return value_head


def create_action_heads(
    action_space: Dict[str, ActionSpace], d_model: int
) -> nn.ModuleDict:
    action_heads = {}
    for name, space in action_space.items():
        action_heads[name] = create_head_for(space, d_model, 16)
    return nn.ModuleDict(action_heads)
