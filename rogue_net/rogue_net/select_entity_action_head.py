import math
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from ragged_buffer import RaggedBufferI64
from torch import nn
from torch.distributions.categorical import Categorical

from entity_gym.env import VecActionMask, VecSelectEntityActionMask
from rogue_net.ragged_tensor import RaggedTensor


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
