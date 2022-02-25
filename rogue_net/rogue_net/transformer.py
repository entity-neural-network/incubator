import logging
from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Optional, Tuple, Union
import math
from entity_gym.environment.environment import ObsSpace
from numpy import dtype
from ragged_buffer import RaggedBufferI64

import torch_scatter

import torch
import torch.nn as nn
from torch.nn import functional as F
from rogue_net.relpos_encoding import RelposEncoding, RelposEncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer network hyperparameters.

    Attributes:
        embd_pdrop: Dropout probability for embedding layer.
        res_pdrop: Dropout probability for residual branches.
        attn_pdrop: Dropout probability for attention.
        nlayer: Number of transformer layers.
        nhead: Number of attention heads.
        dmodel: Dimension of embedding.
        pooling: Replace attention with "mean", "max", or "meanmax" pooling.
        relpos_encoding: Relative positional encoding settings.
    """

    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    n_layer: int = 1
    n_head: int = 2
    d_model: int = 64
    pooling: Optional[Literal["mean", "max", "meanmax"]] = None
    relpos_encoding: Optional[RelposEncodingConfig] = None


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
        self, x: torch.Tensor, batch_index: torch.Tensor, shape: RaggedBufferI64
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
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        shape: RaggedBufferI64,
        relkeysvals: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # For more details on the implementation, see: https://github.com/entity-neural-network/incubator/pull/119
        device = x.device
        padpack = shape.padpack()

        # TODO: only compute indices once
        if padpack is None:
            x = x.reshape(shape.size0(), shape.size1(0), -1)
            attn_mask = None
        else:
            (
                padpack_index,
                padpack_batch,
                padpack_inverse_index,
            ) = padpack
            x = x[torch.LongTensor(padpack_index).to(device)]
            tpadpack_batch = torch.Tensor(padpack_batch).to(device)
            attn_mask = (
                tpadpack_batch.unsqueeze(2) != tpadpack_batch.unsqueeze(1)
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

        # Relative positional encoding (keys)
        if relkeysvals is not None:
            relkeys = relkeysvals[0]  #       (B, T, T, hs)
            # Broadcast and sum over last dimension (dot product of queries with relative keys)
            # TODO: check
            relatt = torch.einsum("bhsd,bstd->bhst", q, relkeys) * (
                1.0 / math.sqrt(k.size(-1))
            )  # (B, nh, T, T)
            att += relatt

        if attn_mask is not None:
            att = att.masked_fill(attn_mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Relative positional encoding (values)
        if relkeysvals is not None:
            relvals = relkeysvals[1]  #       (B, T_query, T_target, hs)
            rely = torch.einsum("bhst,bstd->bhsd", att, relvals)  # (B, nh, T, T)
            y += rely

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if padpack is None:
            return y.reshape(batch_index.size(0), -1)  # type: ignore
        else:
            return y.reshape(y.size(0) * y.size(1), y.size(2))[torch.LongTensor(padpack_inverse_index).to(device)]  # type: ignore


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
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        shape: RaggedBufferI64,
        relkeysvals: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), batch_index, shape, relkeysvals)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, obs_space: ObsSpace) -> None:
        super().__init__()

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        if config.relpos_encoding is not None:
            self.relpos_encoding: Optional[RelposEncoding] = RelposEncoding(
                config.relpos_encoding,
                obs_space,
                dhead=config.d_model // config.n_head,
            )
        else:
            self.relpos_encoding = None

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
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        shape: RaggedBufferI64,
        input_feats: Mapping[str, torch.Tensor],
        index_map: torch.Tensor,
        entity_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.drop(x)

        if self.relpos_encoding is not None:
            device = x.device
            padpack = shape.padpack()
            if padpack is None:
                tpadpack_index = None
            else:
                tpadpack_index = torch.LongTensor(padpack[0]).to(device)
            relkeysvals: Optional[
                Tuple[torch.Tensor, torch.Tensor]
            ] = self.relpos_encoding.keys_values(
                input_feats,
                index_map,
                tpadpack_index,
                shape,
                entity_type,
            )
        else:
            relkeysvals = None

        for block in self.blocks:
            x = block(x, batch_index, shape, relkeysvals)
        return x
