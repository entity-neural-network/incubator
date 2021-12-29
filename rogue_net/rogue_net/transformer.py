import logging
from dataclasses import dataclass
from typing import Literal, Optional, Union
import math
from numpy import dtype
from ragged_buffer import RaggedBufferI64

import torch_scatter

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    n_layer: int = 1
    n_head: int = 1
    d_model: int = 64
    pooling: Optional[Literal["mean", "max", "meanmax"]] = None


class Pool(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.pooling is not None
        # projections
        self.prepool = nn.Linear(config.d_model, config.d_model)
        if config.pooling == "meanmax":
            self.proj = nn.Linear(2 * config.d_model, config.d_model)
        else:
            self.proj = nn.Linear(config.d_model, config.d_model)
        # regularization
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.reduction_op = config.pooling

    def forward(
        self, x: torch.Tensor, batch_index: torch.Tensor, rbatch_index: RaggedBufferI64
    ) -> torch.Tensor:
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


class RaggedAttention(nn.Module):
    """
    A ragged multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.d_model, config.d_model)
        self.query = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head

    def forward(
        self, x: torch.Tensor, batch_index: torch.Tensor, rbatch_index: RaggedBufferI64
    ) -> torch.Tensor:
        device = x.device
        padpack = rbatch_index.padpack()

        if padpack is None:
            x = x.reshape(rbatch_index.size0(), rbatch_index.size1(0), -1)
            attn_mask = None
        else:
            (
                padpack_index,
                padpack_batch,
                padpack_inverse_index,
            ) = padpack
            x = x[torch.LongTensor(padpack_index, device=device)]
            tpadpack_batch = torch.Tensor(padpack_batch, device=device)
            attn_mask = (
                tpadpack_batch.unsqueeze(2) == tpadpack_batch.unsqueeze(1)
            ).unsqueeze(1)

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # full self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if padpack is None:
            return y.reshape(batch_index.size(0), -1)  # type: ignore
        else:
            return y.reshape(y.size(0) * y.size(1), y.size(2))[torch.LongTensor(padpack_inverse_index, device=device)]  # type: ignore


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        if config.pooling is not None:
            self.attn: Union[Pool, RaggedAttention] = Pool(config)
        else:
            self.attn = RaggedAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self, x: torch.Tensor, batch_index: torch.Tensor, rbatch_index: RaggedBufferI64
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), batch_index, rbatch_index)
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

    def forward(
        self, x: torch.Tensor, batch_index: torch.Tensor, rbatch_index: RaggedBufferI64
    ) -> torch.Tensor:
        # position_embeddings = self.pos_emb[
        #    :, :t, :
        # ]  # each position maps to a (learnable) vector
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, batch_index, rbatch_index)
        return x
