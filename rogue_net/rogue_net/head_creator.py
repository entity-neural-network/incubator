import math
from typing import Dict, Optional, Tuple
from ragged_buffer import RaggedBufferI64
from rogue_net.ragged_tensor import RaggedTensor
from torch import nn
from torch.distributions.categorical import Categorical
import torch
import numpy as np
import numpy.typing as npt
from entity_gym.environment import (
    VecActionMask,
    ActionSpace,
    VecCategoricalActionMask,
    CategoricalActionSpace,
    SelectEntityActionMask,
    VecSelectEntityActionMask,
    SelectEntityActionSpace,
)
from typing import Dict


class ActionHead(nn.Module):
    def forward(
        self,
        x: RaggedTensor,
        index_offsets: RaggedBufferI64,
        mask: VecActionMask,
        prev_actions: Optional[RaggedBufferI64],
    ) -> Tuple[
        RaggedBufferI64,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        mask: VecActionMask,
        prev_actions: Optional[RaggedBufferI64],
    ) -> Tuple[
        torch.Tensor, npt.NDArray[np.int64], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        assert isinstance(
            mask, VecCategoricalActionMask
        ), f"Expected CategoricalActionMaskBatch, got {type(mask)}"

        device = x.data.device
        lengths = mask.actors.size1()
        if len(mask.actors) == 0:
            return (
                torch.zeros((0), dtype=torch.int64, device=device),
                lengths,
                torch.zeros((0), dtype=torch.float32, device=device),
                torch.zeros((0), dtype=torch.float32, device=device),
                torch.zeros((0, self.n_choice), dtype=torch.float32, device=device),
            )

        actors = (
            torch.tensor((mask.actors + index_offsets).as_array())
            .to(x.data.device)
            .squeeze(-1)
        )
        actor_embeds = x.data[actors]
        logits = self.proj(actor_embeds)

        # Apply masks from the environment
        if mask.mask is not None and mask.mask.size0() > 0:
            reshaped_masks = torch.tensor(
                mask.mask.as_array().reshape(logits.shape)
            ).to(x.data.device)
            logits = logits.masked_fill(reshaped_masks == 0, -float("inf"))

        dist = Categorical(logits=logits)
        if prev_actions is None:
            action = dist.sample()
        else:
            action = torch.tensor(prev_actions.as_array().squeeze(-1)).to(x.data.device)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, lengths, logprob, entropy, dist.logits


class PaddedSelectEntityActionHead(nn.Module):
    """
    Action head for selecting entities.
    See https://github.com/entity-neural-network/incubator/pull/109 for more details.
    """

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
        mask: VecActionMask,
        prev_actions: Optional[RaggedBufferI64],
    ) -> Tuple[
        torch.Tensor, npt.NDArray[np.int64], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        assert isinstance(
            mask, VecSelectEntityActionMask
        ), f"Expected SelectEntityActionMaskBatch, got {type(mask)}"
        device = x.data.device
        actor_lengths = mask.actors.size1()
        if len(mask.actors) == 0:
            return (
                torch.zeros((0), dtype=torch.int64, device=device),
                actor_lengths,
                torch.zeros((0), dtype=torch.float32, device=device),
                torch.zeros((0), dtype=torch.float32, device=device),
                torch.zeros((0), dtype=torch.float32, device=device),
            )

        actors = torch.tensor(
            (mask.actors + index_offsets).as_array(), device=device
        ).squeeze(-1)
        actor_embeds = x.data[actors]
        queries = self.query_proj(actor_embeds).squeeze(1)
        max_actors = actor_lengths.max()
        # TODO: can omit rows that are only padding
        padded_queries = torch.zeros(
            len(actor_lengths) * max_actors, self.d_qk, device=device
        )
        qindices = torch.tensor(
            (mask.actors.indices(1) + mask.actors.indices(0) * max_actors)
            .as_array()
            .flatten(),
            device=device,
        )
        padded_queries[qindices] = queries
        padded_queries = padded_queries.view(len(actor_lengths), max_actors, self.d_qk)
        query_mask = torch.zeros(len(actor_lengths) * max_actors, device=device)
        query_mask[qindices] = 1
        query_mask = query_mask.view(len(actor_lengths), max_actors)

        actee_lengths = mask.actees.size1()
        actees = torch.tensor(
            (mask.actees + index_offsets).as_array(), device=device
        ).squeeze(-1)
        actee_embeds = x.data[actees]
        keys = self.key_proj(actee_embeds).squeeze(1)
        max_actees = actee_lengths.max()
        padded_keys = torch.ones(
            len(actee_lengths) * max_actees, self.d_qk, device=device
        )
        kindices = torch.tensor(
            (mask.actees.indices(1) + mask.actees.indices(0) * max_actees)
            .as_array()
            .flatten(),
            device=device,
        )
        padded_keys[kindices] = keys
        padded_keys = padded_keys.view(len(actee_lengths), max_actees, self.d_qk)
        key_mask = torch.zeros(len(actee_lengths) * max_actees, device=device)
        key_mask[kindices] = 1
        key_mask = key_mask.view(len(actee_lengths), max_actees)

        logits = torch.bmm(padded_queries, padded_keys.transpose(1, 2)) * (
            1.0 / math.sqrt(self.d_qk)
        )
        logits_mask = torch.bmm(query_mask.unsqueeze(2), key_mask.unsqueeze(1))

        # Firstly mask off the conditions that are not available. This is the typical masked transformer approach
        logits = logits.masked_fill(logits_mask == 0, -1e9)

        dist = Categorical(logits=logits)
        if prev_actions is None:
            action = dist.sample()
        else:
            action = torch.tensor(prev_actions.as_array(), device=device).flatten()
            padded_actions = torch.ones(
                (logits.size(0) * logits.size(1)), dtype=torch.long, device=device
            )
            padded_actions[qindices] = action
            action = padded_actions.view(len(actor_lengths), max_actors)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return (
            action.flatten()[qindices],
            actor_lengths,
            logprob.flatten()[qindices],
            entropy.flatten()[qindices],
            logits,
        )


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
    action_space: Dict[str, ActionSpace], d_model: int, d_qk: int
) -> nn.ModuleDict:
    action_heads = {}
    for name, space in action_space.items():
        action_heads[name] = create_head_for(space, d_model, d_qk)
    return nn.ModuleDict(action_heads)
