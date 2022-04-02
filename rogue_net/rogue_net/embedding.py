from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import ragged_buffer
import torch
from ragged_buffer import RaggedBufferF32, RaggedBufferI64
from torch import nn

from entity_gym.environment import ObsSpace
from entity_gym.simple_trace import Tracer
from rogue_net.input_norm import InputNorm
from rogue_net.translate_positions import TranslatePositions, TranslationConfig


class EntityEmbedding(nn.Module):
    def __init__(
        self,
        obs_space: ObsSpace,
        feature_transforms: Optional[TranslationConfig],
        d_model: int,
    ) -> None:
        super().__init__()
        if feature_transforms is not None:
            self.feature_transforms: Optional[TranslatePositions] = TranslatePositions(
                feature_transforms, obs_space
            )
            obs_space = self.feature_transforms.transform_obs_space(obs_space)
        else:
            self.feature_transforms = None

        embeddings: Dict[str, nn.Module] = {}
        for name, entity in obs_space.entities.items():
            if entity.features:
                embeddings[name] = nn.Sequential(
                    InputNorm(len(entity.features)),
                    nn.Linear(len(entity.features), d_model),
                    nn.ReLU(),
                    nn.LayerNorm(d_model),
                )
            else:
                embeddings[name] = FeaturelessEmbedding(d_model)
        self.embeddings = nn.ModuleDict(embeddings)

    def forward(
        self,
        entities: Mapping[str, RaggedBufferF32],
        tracer: Tracer,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        RaggedBufferI64,
        Mapping[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        expected_order = iter(self.embeddings.keys())
        for name in entities.keys():
            while True:
                try:
                    entity = next(expected_order)
                except StopIteration:
                    raise ValueError(
                        f"Fatal error: order of entities in the input does not match the order of observation space: {list(entities.keys())} != {list(self.embeddings.keys())}"
                    )
                if entity == name:
                    break

        entity_embeds = []
        index_offsets = {}
        index_offset = 0
        entity_type = []

        if self.feature_transforms:
            entities = {name: feats.clone() for name, feats in entities.items()}
            self.feature_transforms.apply(entities)
        tentities = {
            entity: torch.tensor(features.as_array(), device=device)
            for entity, features in entities.items()
        }

        for i, (entity, embedding) in enumerate(self.embeddings.items()):
            # We may have environment states that do not contain every possible entity
            if entity in entities:
                batch = tentities[entity]
                emb = embedding(batch)
                entity_embeds.append(emb)
                entity_type.append(
                    torch.full((emb.size(0), 1), float(i), device=device)
                )
                index_offsets[entity] = index_offset
                index_offset += batch.size(0)

        x = torch.cat(entity_embeds)
        with tracer.span("ragged_metadata"):
            lengths = sum(entity.size1() for entity in entities.values())
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
            tindex_map = torch.tensor(index_map.as_array().flatten(), device=device)
            tbatch_index = torch.tensor(batch_index, device=device)
            tlengths = torch.tensor(lengths, device=device)

        x = x[tindex_map]
        entity_types = torch.cat(entity_type)[tindex_map]
        tbatch_index = tbatch_index[tindex_map]

        return (
            x,
            tbatch_index,
            index_map,
            tentities,
            tindex_map,
            entity_types,
            tlengths,
        )


class FeaturelessEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Parameter(torch.randn(1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.repeat(x.size(0), 1)
