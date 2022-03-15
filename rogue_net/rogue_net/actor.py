from typing import Dict, List, Mapping, Optional, Tuple, Type, TypeVar

from entity_gym.environment import (
    ActionMaskBatch,
    ActionSpace,
    ObsSpace,
)
from enn_ppo.simple_trace import Tracer
import numpy as np
import ragged_buffer
from rogue_net.ragged_tensor import RaggedTensor
import torch
import torch.nn as nn
import torch_scatter
import numpy.typing as npt

from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBuffer
import rogue_net.head_creator as head_creator
import rogue_net.embedding_creator as embedding_creator
from rogue_net.transformer import Transformer, TransformerConfig


ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)


def tensor_dict_to_ragged(
    rb_cls: Type[RaggedBuffer[ScalarType]],
    d: Dict[str, torch.Tensor],
    lengths: Dict[str, npt.NDArray[np.int64]],
) -> Dict[str, RaggedBuffer[ScalarType]]:
    return {k: rb_cls.from_flattened(v.cpu().numpy(), lengths[k]) for k, v in d.items()}


class Actor(nn.Module):
    def __init__(
        self,
        obs_space: ObsSpace,
        embedding: nn.ModuleDict,
        action_space: Dict[str, ActionSpace],
        backbone: nn.Module,
        action_heads: nn.ModuleDict,
        auxiliary_heads: Optional[nn.ModuleDict] = None,
    ):
        super(Actor, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.embedding = embedding
        self.backbone = backbone
        self.action_heads = action_heads
        self.auxiliary_heads = auxiliary_heads

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def batch_and_embed(
        self, entities: Mapping[str, RaggedBufferF32], tracer: Tracer
    ) -> RaggedTensor:
        entity_embeds = []
        index_offsets = {}
        index_offset = 0

        for entity, embedding in self.embedding.items():
            # We may have environment states that do not contain every possible entity
            if entity in entities:
                batch = torch.tensor(entities[entity].as_array()).to(self.device())
                entity_embeds.append(embedding(batch))
                index_offsets[entity] = index_offset
                index_offset += batch.size(0)
        x = torch.cat(entity_embeds)
        with tracer.span("ragged_metadata"):
            lengths = sum([entity.size1() for entity in entities.values()])
            batch_index = np.concatenate(
                [entity.indices(0).as_array().flatten() for entity in entities.values()]
            )
            index_map = (
                ragged_buffer.cat(
                    [
                        entity.flat_indices() + index_offsets[name]
                        for name, entity in entities.items()
                    ],
                    dim=1,
                )
                .as_array()
                .flatten()
            )
            tindex_map = torch.tensor(index_map).to(self.device())
            tbatch_index = torch.tensor(batch_index).to(self.device())
            tlengths = torch.tensor(lengths).to(self.device())

        with tracer.span("backbone"):
            x = self.backbone(x, tbatch_index)

        return RaggedTensor(
            x,
            tindex_map,
            tbatch_index,
            tlengths,
        )

    def get_auxiliary_head(
        self, entities: Mapping[str, RaggedBufferF32], head_name: str, tracer: Tracer
    ) -> torch.Tensor:
        x = self.batch_and_embed(entities, tracer)
        pooled = torch_scatter.scatter(
            src=x.data, dim=0, index=x.batch_index, reduce="mean"
        )
        return self.auxiliary_heads[head_name](pooled)  # type: ignore

    def get_action_and_auxiliary(
        self,
        entities: Mapping[str, RaggedBufferF32],
        action_masks: Mapping[str, ActionMaskBatch],
        tracer: Tracer,
        prev_actions: Optional[Dict[str, RaggedBufferI64]] = None,
    ) -> Tuple[
        Dict[str, RaggedBufferI64],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, npt.NDArray[np.int64]],
        Dict[str, torch.Tensor],
    ]:
        actions = {}
        probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}
        with tracer.span("batch_and_embed"):
            x = self.batch_and_embed(entities, tracer)

        tracer.start("action_heads")
        index_offsets = RaggedBufferI64.from_array(
            torch.cat([torch.tensor([0]).to(self.device()), x.lengths[:-1]])
            .cumsum(0)
            .cpu()
            .numpy()
            .reshape(-1, 1, 1)
        )
        actor_counts: Dict[str, np.ndarray] = {}
        for action_name, action_head in self.action_heads.items():
            action, count, logprob, entropy = action_head(
                x,
                index_offsets,
                action_masks[action_name],
                prev_actions[action_name] if prev_actions is not None else None,
            )
            actor_counts[action_name] = count
            actions[action_name] = action
            probs[action_name] = logprob
            entropies[action_name] = entropy
        tracer.end("action_heads")

        tracer.start("auxiliary_heads")
        if self.auxiliary_heads:
            pooled = torch_scatter.scatter(
                src=x.data, dim=0, index=x.batch_index, reduce="mean"
            )
            auxiliary_values = {
                name: module(pooled) for name, module in self.auxiliary_heads.items()
            }
        else:
            auxiliary_values = {}
        tracer.end("auxiliary_heads")

        return (
            prev_actions
            or tensor_dict_to_ragged(RaggedBufferI64, actions, actor_counts),
            probs,
            entropies,
            actor_counts,
            auxiliary_values,
        )


class AutoActor(Actor):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        d_model: int,
        d_qk: int = 16,
        auxiliary_heads: Optional[nn.ModuleDict] = None,
        n_layer: int = 1,
    ):
        self.d_model = d_model
        super().__init__(
            obs_space,
            embedding_creator.create_embeddings(obs_space, d_model),
            action_space,
            Transformer(TransformerConfig(d_model=d_model, n_layer=n_layer)),
            head_creator.create_action_heads(action_space, d_model, d_qk),
            auxiliary_heads=auxiliary_heads,
        )
