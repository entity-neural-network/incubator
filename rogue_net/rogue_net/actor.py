from typing import Dict, Mapping, Optional, Tuple, Type, TypeVar

from entity_gym.environment import (
    ActionMaskBatch,
    ActionSpace,
    ObsSpace,
)
from enn_ppo.simple_trace import Tracer
import numpy as np
import ragged_buffer
from rogue_net.relpos_encoding import (
    RelposEncoding,
    RelposEncodingConfig,
)
from rogue_net.ragged_tensor import RaggedTensor
from rogue_net.translate_positions import TranslatePositions
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
        feature_transforms: Optional[TranslatePositions] = None,
        relpos_encoding: Optional[RelposEncoding] = None,
    ):
        super(Actor, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.embedding = embedding
        self.backbone = backbone
        self.action_heads = action_heads
        self.auxiliary_heads = auxiliary_heads
        self.feature_transforms = feature_transforms
        self.relpos_encoding = True

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def batch_and_embed(
        self, entities: Mapping[str, RaggedBufferF32], tracer: Tracer
    ) -> RaggedTensor:
        entity_embeds = []
        index_offsets = {}
        index_offset = 0
        entity_type = []

        if self.feature_transforms:
            entities = {name: feats.clone() for name, feats in entities.items()}
            self.feature_transforms.apply(entities)
        tentities = {
            entity: torch.tensor(features.as_array()).to(self.device())
            for entity, features in entities.items()
        }

        for i, (entity, embedding) in enumerate(self.embedding.items()):
            # We may have environment states that do not contain every possible entity
            if entity in entities:
                batch = tentities[entity]
                emb = embedding(batch)
                entity_embeds.append(emb)
                entity_type.append(
                    torch.full((emb.size(0), 1), float(i)).to(self.device())
                )
                index_offsets[entity] = index_offset
                index_offset += batch.size(0)

        x = torch.cat(entity_embeds)
        with tracer.span("ragged_metadata"):
            lengths = sum([entity.size1() for entity in entities.values()])
            batch_index = np.concatenate(
                [entity.indices(0).as_array().flatten() for entity in entities.values()]
            )
            index_map = ragged_buffer.cat(
                [
                    entity.flat_indices() + index_offsets[name]
                    for name, entity in entities.items()
                ],
                dim=1,
            )
            tindex_map = torch.tensor(index_map.as_array().flatten()).to(self.device())
            tbatch_index = torch.tensor(batch_index).to(self.device())
            tlengths = torch.tensor(lengths).to(self.device())

        x = x[tindex_map]
        entity_types = torch.cat(entity_type)[tindex_map]
        tbatch_index = tbatch_index[tindex_map]

        with tracer.span("backbone"):
            x = self.backbone(
                x, tbatch_index, index_map, tentities, tindex_map, entity_types
            )

        return RaggedTensor(
            x,
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
        Dict[str, RaggedBufferI64],  # actions
        Dict[str, torch.Tensor],  # action probabilities
        Dict[str, torch.Tensor],  # entropy
        Dict[str, npt.NDArray[np.int64]],  # number of actors in each frame
        Dict[str, torch.Tensor],  # auxiliary head values
        Dict[str, torch.Tensor],  # full logits
    ]:
        actions = {}
        probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
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
            action, count, logprob, entropy, logit = action_head(
                x,
                index_offsets,
                action_masks[action_name],
                prev_actions[action_name] if prev_actions is not None else None,
            )
            actor_counts[action_name] = count
            actions[action_name] = action
            probs[action_name] = logprob
            entropies[action_name] = entropy
            if logit is not None:
                logits[action_name] = logit

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
            logits,
        )


class AutoActor(Actor):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        d_model: int,
        n_head: int,
        d_qk: int = 16,
        auxiliary_heads: Optional[nn.ModuleDict] = None,
        n_layer: int = 1,
        pooling_op: Optional[str] = None,
        feature_transforms: Optional[TranslatePositions] = None,
        relpos_encoding: Optional[RelposEncodingConfig] = None,
    ):
        assert pooling_op in (None, "mean", "max", "meanmax")
        self.d_model = d_model
        super().__init__(
            obs_space,
            embedding_creator.create_embeddings(obs_space, d_model),
            action_space,
            Transformer(
                TransformerConfig(
                    d_model=d_model,
                    n_head=n_head,
                    n_layer=n_layer,
                    pooling=pooling_op,  # type: ignore
                    relpos_encoding=relpos_encoding,
                )
            ),
            head_creator.create_action_heads(action_space, d_model, d_qk),
            auxiliary_heads=auxiliary_heads,
            feature_transforms=feature_transforms,
            relpos_encoding=RelposEncoding(relpos_encoding)
            if relpos_encoding
            else None,
        )
