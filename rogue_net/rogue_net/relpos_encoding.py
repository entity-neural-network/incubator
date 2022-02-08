from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple, overload
import torch.nn as nn
import torch
from ragged_buffer import RaggedBufferI64

from entity_gym.environment import ObsSpace


@dataclass(frozen=True)
class RelposEncodingConfig:
    extent: List[int]
    position_features: List[str]
    obs_space: ObsSpace
    d_head: int
    per_entity_values: bool = True
    exclude_entities: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert len(self.extent) == len(self.position_features)


class RelposEncoding(nn.Module):
    def __init__(
        self,
        config: RelposEncodingConfig,
    ) -> None:
        super().__init__()
        self.d_head = config.d_head
        self.positional_features = config.position_features
        self.n_entity = len(config.obs_space.entities)
        self.per_entity_values = config.per_entity_values
        self.exclude_entities = config.exclude_entities
        strides = []
        positions = 1
        for extent in config.extent:
            strides.append(float(positions))
            positions *= 2 * extent + 1
        self.positions = positions
        self.register_buffer("strides", torch.tensor(strides).unsqueeze(0))
        self.register_buffer("extent", torch.tensor(config.extent).view(1, 1, 1, -1))
        # TODO: tune embedding init scale
        self.keys = nn.Embedding(self.positions, self.d_head)
        self.values = nn.Embedding(
            self.positions * self.n_entity
            if config.per_entity_values
            else self.positions,
            self.d_head,
        )
        self.keys.weight.data.normal_(mean=0.0, std=0.05)
        self.values.weight.data.normal_(mean=0.0, std=0.2)
        self.position_feature_indices = {
            entity_name: torch.LongTensor(
                [
                    entity.features.index(feature_name)
                    for feature_name in config.position_features
                ]
            )
            for entity_name, entity in config.obs_space.entities.items()
            if entity_name not in self.exclude_entities
        }

    def keys_values(
        self,
        # Dict from entity name to raw input features
        x: Mapping[str, torch.Tensor],
        # Maps entities ordered first by type to entities ordered first by frame
        index_map: torch.Tensor,
        # Maps flattened embeddings to packed/padded tensor with fixed sequence lengths
        packpad_index: Optional[torch.Tensor],
        # Ragged shape of the flattened embeddings tensor
        shape: RaggedBufferI64,
        # Type of each entity
        entity_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = []
        for entity_name, features in x.items():
            if entity_name not in self.exclude_entities:
                positions.append(
                    features[:, self.position_feature_indices[entity_name]]
                )
        # Flat tensor of positions
        tpos = torch.cat(positions, dim=0)
        # Flat tensor of positions ordered by sample
        tpos = tpos[index_map]
        # Padded/packed Batch x Seq x Pos tensor of positions
        if packpad_index is not None:
            tpos = tpos[packpad_index]
            entity_type = entity_type[packpad_index]
        else:
            tpos = tpos.reshape(shape.size0(), shape.size1(0), -1)
            entity_type = entity_type.reshape(shape.size0(), shape.size1(0), 1)

        # Batch x Seq x Seq x Pos relative positions
        relative_positions = tpos.unsqueeze(1) - tpos.unsqueeze(2)

        clamped_positions = torch.max(
            torch.min(
                self.extent,  # type: ignore
                relative_positions,
            ),
            -self.extent,  # type: ignore
        )
        positive_positions = clamped_positions + self.extent
        indices = (positive_positions * self.strides).sum(dim=-1).long()

        # Batch x Seq x Seq x d_model
        keys = self.keys(indices)

        if self.per_entity_values:
            per_entity_type_indices = indices + (
                entity_type * self.positions
            ).transpose(2, 1).long().repeat(1, indices.size(2), 1)
        else:
            per_entity_type_indices = indices
        values = self.values(per_entity_type_indices)

        return keys, values
