from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from ragged_buffer import RaggedBufferI64
from torch import nn
from torch.distributions.categorical import Categorical

from entity_gym.env import VecActionMask, VecCategoricalActionMask
from rogue_net.ragged_tensor import RaggedTensor


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


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)  # type: ignore
    return layer
