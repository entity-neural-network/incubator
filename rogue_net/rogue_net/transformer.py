import logging
from dataclasses import dataclass
from typing import Literal

import torch_scatter

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    n_layer: int = 1
    n_head: int = 1
    d_model: int = 64
    pooling: Literal["mean", "max", "meanmax"] = "mean"


class Pool(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # projections
        self.prepool = nn.Linear(config.d_model, config.d_model)
        if config.pooling == "meanmax":
            self.proj = nn.Linear(2 * config.d_model, config.d_model)
        else:
            self.proj = nn.Linear(config.d_model, config.d_model)
        # regularization
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.reduction_op = config.pooling

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        x = self.prepool(x)

        if "mean" in self.reduction_op:
            xmean = torch_scatter.scatter(
                src=x, dim=0, index=batch_index, reduce="mean"
            )
            xpool = xmean
        if "max" in self.reduction_op:
            xmax = torch_scatter.scatter(src=x, dim=0, index=batch_index, reduce="max")
            xpool = xmax
        if "meanmax" in self.reduction_op:
            xpool = torch.cat([xmean, xmax], dim=1)
        x = self.proj(xpool)

        return self.resid_drop(x[batch_index])  # type: ignore


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = Pool(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), batch_index)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.d_model))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        # position_embeddings = self.pos_emb[
        #    :, :t, :
        # ]  # each position maps to a (learnable) vector
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, batch_index)
        return x
