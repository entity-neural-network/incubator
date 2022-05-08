import logging
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_scatter
from ragged_buffer import RaggedBufferI64
from torch.nn import functional as F

from entity_gym.env.environment import ObsSpace
from rogue_net.relpos_encoding import RelposEncoding, RelposEncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer network hyperparameters.

    Attributes:
        embd_pdrop: Dropout probability for embedding layer.
        resid_pdrop: Dropout probability for residual branches.
        attn_pdrop: Dropout probability for attention.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        d_model: Dimension of embedding.
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

    def __init__(
        self, config: TransformerConfig, relpos_encoding: Optional[RelposEncoding]
    ) -> None:
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
        self.relpos_encoding = relpos_encoding

    def forward(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        shape: RaggedBufferI64,
        visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # For more details on the implementation, see: https://github.com/entity-neural-network/incubator/pull/119
        device = x.device
        padpack = shape.padpack()

        # TODO: only compute indices once
        if padpack is None:
            nbatch = shape.size0()
            nseq = shape.size1(0) if shape.items() > 0 else 0
            x = x.reshape(nbatch, nseq, x.size(-1))
            if visible is not None:
                attn_mask: Optional[torch.Tensor] = (
                    visible.reshape(nbatch, nseq, 1) > visible.reshape(nbatch, 1, nseq)
                ).unsqueeze(1)
            else:
                attn_mask = None
        else:
            (
                padpack_index,
                padpack_batch,
                padpack_inverse_index,
            ) = padpack
            tpadpack_index = torch.tensor(
                padpack_index, dtype=torch.long, device=device
            )
            x = x[tpadpack_index]
            tpadpack_batch = torch.tensor(padpack_batch, device=device)
            attn_mask = (
                tpadpack_batch.unsqueeze(2) != tpadpack_batch.unsqueeze(1)
            ).unsqueeze(1)
            if visible is not None:
                visible = visible[tpadpack_index]
                attn_mask.logical_or_(
                    (visible.unsqueeze(2) > visible.unsqueeze(1)).unsqueeze(1)
                )

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
        if self.relpos_encoding is not None:
            att += self.relpos_encoding.relattn_logits(q)

        if attn_mask is not None:
            att = att.masked_fill(attn_mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Relative positional encoding (values)
        if self.relpos_encoding is not None:
            y += self.relpos_encoding.relpos_values(att, x)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if padpack is None:
            return y.reshape(batch_index.size(0), y.size(-1))  # type: ignore
        else:
            return y.reshape(y.size(0) * y.size(1), y.size(2))[torch.tensor(padpack_inverse_index, dtype=torch.long, device=device)]  # type: ignore


class Block(nn.Module):
    def __init__(
        self, config: TransformerConfig, relpos_encoding: Optional[RelposEncoding]
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        if config.pooling is not None:
            self.attn: Union[Pool, RaggedAttention] = Pool(config)
        else:
            self.attn = RaggedAttention(config, relpos_encoding)
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
        visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), batch_index, shape, visible)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, obs_space: ObsSpace) -> None:
        super().__init__()

        if config.relpos_encoding is not None:
            self.relpos_encoding: Optional[RelposEncoding] = RelposEncoding(
                config.relpos_encoding,
                obs_space,
                dmodel=config.d_model,
                dhead=config.d_model // config.n_head,
            )
        else:
            self.relpos_encoding = None
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[Block(config, self.relpos_encoding) for _ in range(config.n_layer)]
        )

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
        visible: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.drop(x)

        if self.relpos_encoding is not None:
            device = x.device
            padpack = shape.padpack()
            if padpack is None:
                tpadpack_index = None
            else:
                tpadpack_index = torch.tensor(
                    padpack[0], dtype=torch.long, device=device
                )
            relkeysvals: Optional[
                Tuple[torch.Tensor, torch.Tensor]
            ] = self.relpos_encoding.keys_values(
                input_feats,
                index_map,
                tpadpack_index,
                shape,
                entity_type,
            )
            self.relpos_encoding.cached_rkvs = relkeysvals
        else:
            relkeysvals = None

        for block in self.blocks:
            x = block(x, batch_index, shape, visible)
        return x
