from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple, overload
import torch.nn as nn
import torch
from ragged_buffer import RaggedBufferI64

from entity_gym.environment import ObsSpace


@dataclass(frozen=True, eq=False)
class RelposEncodingConfig:
    """Settings for relative positio encoding.

    Attributes:
        extent: Each integer relative position in the interval [-extent, extent] receives a positional embedding, with positions outside the interval snapped to the closest end.
        position_features: Names of position features used for relative position encoding.
        scale: Relative positions are divided by the scale before being assigned an embedding.
        per_entity_values: Whether to use per-entity embeddings for relative positional values.
        exclude_entities: List of entity types to exclude from relative position encoding.
    """

    extent: List[int]
    position_features: List[str]
    scale: float = 1.0
    per_entity_values: bool = True
    exclude_entities: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert len(self.extent) == len(self.position_features)


class RelposEncoding(nn.Module, RelposEncodingConfig):
    def __init__(
        self, config: RelposEncodingConfig, obs_space: ObsSpace, dhead: int
    ) -> None:
        nn.Module.__init__(self)
        RelposEncodingConfig.__init__(self, **config.__dict__)

        self.n_entity = len(obs_space.entities)
        strides = []
        positions = 1
        for extent in self.extent:
            strides.append(float(positions))
            positions *= 2 * extent + 1
        self.positions = positions
        self.register_buffer("strides", torch.tensor(strides).unsqueeze(0))
        self.register_buffer(
            "extent_tensor", torch.tensor(self.extent).view(1, 1, 1, -1)
        )
        # TODO: tune embedding init scale
        self.keys = nn.Embedding(self.positions, dhead)
        self.values = nn.Embedding(
            self.positions * self.n_entity
            if config.per_entity_values
            else self.positions,
            dhead,
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
            for entity_name, entity in obs_space.entities.items()
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
        relative_positions = (tpos.unsqueeze(1) - tpos.unsqueeze(2)) * (
            1.0 / self.scale
        )

        clamped_positions = torch.max(
            torch.min(
                self.extent_tensor,  # type: ignore
                relative_positions,
            ),
            -self.extent_tensor,  # type: ignore
        )
        positive_positions = clamped_positions + self.extent_tensor
        indices = (positive_positions * self.strides).sum(dim=-1).round().long()

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
