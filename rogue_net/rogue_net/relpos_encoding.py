import math
from dataclasses import dataclass, field
from typing import List, Literal, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ragged_buffer import RaggedBufferI64

from entity_gym.env import ObsSpace
from rogue_net.input_norm import InputNorm


@dataclass(frozen=True, eq=False)
class RelposEncodingConfig:
    """Settings for relative position encoding.

    Attributes:
        extent: Each integer relative position in the interval [-extent, extent] receives a positional embedding, with positions outside the interval snapped to the closest end.
        position_features: Names of position features used for relative position encoding.
        scale: Relative positions are divided by the scale before being assigned an embedding.
        per_entity_values: Whether to use per-entity embeddings for relative positional values.
        exclude_entities: List of entity types to exclude from relative position encoding.
        key_relpos_projection: Adds a learnable projection from the relative position/distance to the relative positional keys.
        value_relpos_projection: Adds a learnable projection from the relative position/distance to the relative positional values.
        per_entity_projections: Uses a different learned projection per entity type for the `key_relpos_projection` and `value_relpos_projection`.
        radial: Buckets all relative positions by their angle. The `extent` is interpreted as the number of buckets.
        distance: Buckets all relative positions by their distance. The `extent` is interpreted as the number of buckets.
        rotation_vec_features: Name of features that give a unit orientation vector for each entity by which to rotate relative positions.
        rotation_angle_feature: Name of feature that gives an angle in radians by which to rotate relative positions.
        interpolate: Whether to interpolate between the embeddings of neighboring positions.
    """

    extent: List[int]
    position_features: List[str]
    scale: float = 1.0
    per_entity_values: bool = False
    exclude_entities: List[str] = field(default_factory=list)
    value_relpos_projection: bool = False
    key_relpos_projection: bool = False
    per_entity_projections: bool = False
    radial: bool = False
    distance: bool = False
    rotation_vec_features: Optional[List[str]] = None
    rotation_angle_feature: Optional[str] = None
    interpolate: bool = False
    value_gate: Literal["linear", "relu", "gelu", "sigmoid", None] = "relu"

    def __post_init__(self) -> None:
        if self.radial and self.distance:
            assert (
                len(self.extent) == 2
            ), "Polar relative position encoding expects two extent values (number of angle buckets and number of distance buckets)"
        elif self.radial:
            assert (
                len(self.extent) == 1
            ), "Radial relative position encoding expects a single extent value (number of angle buckets)"
            assert (
                self.rotation_angle_feature is not None
                or self.rotation_vec_features is not None
            ), "Radial relative position encoding requires `rotation_angle_feature` or `rotation_vec_features` to be set"
        elif self.distance:
            assert (
                len(self.extent) == 1
            ), "Distance relative position encoding expects a single extent value (number of distance buckets)"
        else:
            assert len(self.extent) == len(
                self.position_features
            ), "Relative position encoding expects a extent value for each position feature"
        assert (
            self.rotation_vec_features is None or self.rotation_angle_feature is None
        ), "Only one of rotation_vec_features and rotation_angle_feature can be specified"


class RelposEncoding(nn.Module, RelposEncodingConfig):
    def __init__(
        self, config: RelposEncodingConfig, obs_space: ObsSpace, dmodel: int, dhead: int
    ) -> None:
        nn.Module.__init__(self)
        RelposEncodingConfig.__init__(self, **config.__dict__)

        self.n_entity = len(obs_space.entities)
        if self.radial and self.distance:
            angles, distances = self.extent
            strides = [1.0, angles]
            positions = angles * distances
        elif self.radial or self.distance:
            strides = [1.0]
            positions = self.extent[0]
        else:
            strides = []
            positions = 1
            for extent in self.extent:
                strides.append(float(positions))
                positions *= 2 * extent + 1
        self.positions = positions
        self.register_buffer("strides", torch.tensor(strides).unsqueeze(0))
        self.register_buffer(
            "extent_tensor", torch.tensor(self.extent).view(1, 1, 1, -1).long()
        )
        # TODO: tune embedding init scale
        self.keys = nn.Embedding(self.positions, dhead)
        self.values = nn.Embedding(
            self.positions * self.n_entity
            if config.per_entity_values
            else self.positions,
            dhead,
        )
        self.distance_values = nn.Embedding(self.n_entity, dhead)
        self.keys.weight.data.normal_(mean=0.0, std=0.05)
        self.values.weight.data.normal_(mean=0.0, std=0.2)
        self.distance_values.weight.data.normal_(mean=0.0, std=0.2)
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
        self.orientation_vec_indices = (
            {
                entity_name: torch.LongTensor(
                    [
                        entity.features.index(feature_name)
                        for feature_name in self.rotation_vec_features
                    ]
                )
                for entity_name, entity in obs_space.entities.items()
                if entity_name not in self.exclude_entities
                and all(
                    feature in entity.features for feature in self.rotation_vec_features
                )
            }
            if self.rotation_vec_features is not None
            else None
        )
        self.orientation_angle_index = (
            {
                entity_name: entity.features.index(self.rotation_angle_feature)
                for entity_name, entity in obs_space.entities.items()
                if entity_name not in self.exclude_entities
                and self.rotation_angle_feature in entity.features
            }
            if self.rotation_angle_feature is not None
            else None
        )
        if self.value_relpos_projection:
            if self.per_entity_projections:
                self.vproj: nn.Module = nn.ModuleDict(
                    {
                        entity_name: nn.Linear(3, dhead)
                        for entity_name in self.position_feature_indices
                    }
                )
            else:
                self.vproj = nn.Linear(3, dhead)
        if self.key_relpos_projection:
            if self.per_entity_projections:
                self.kproj: nn.Module = nn.ModuleDict(
                    {
                        entity_name: nn.Linear(3, dhead)
                        for entity_name in self.position_feature_indices
                    }
                )
            else:
                self.kproj = nn.Linear(3, dhead)
        if self.key_relpos_projection or self.value_relpos_projection:
            self.relpos_norm = InputNorm(3)
        if self.value_gate is not None:
            self.value_gate_proj = nn.Linear(dmodel, dhead)
        self.cached_rkvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.global_entity = "__global__" in obs_space.entities

    def relattn_logits(self, queries: torch.Tensor) -> torch.Tensor:
        assert self.cached_rkvs is not None
        relkeys = self.cached_rkvs[0]  #       (B, T, T, dhead)
        # Broadcast and sum over last dimension (dot product of queries with relative keys)
        relattn: torch.Tensor = torch.einsum("bhsd,bstd->bhst", queries, relkeys) * (
            1.0 / math.sqrt(relkeys.size(-1))
        )  # (B, nh, T, T)
        return relattn

    def relpos_values(self, att: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert self.cached_rkvs is not None
        relvals = self.cached_rkvs[1]  #       (B, T_query, T_target, dhead)
        if self.value_gate is not None:
            vgate = self.value_gate_proj(x)  # (B, T_target, dhead)
            if self.value_gate == "relu":
                vgate = F.relu(vgate)
            elif self.value_gate == "gelu":
                vgate = F.gelu(vgate)
            elif self.value_gate == "sigmoid":
                vgate = torch.sigmoid(vgate)

            relvals = torch.einsum("bqtd,btd->bqtd", relvals, vgate)
        rely: torch.Tensor = torch.einsum(
            "bhst,bstd->bhsd", att, relvals
        )  # (B, nh, T, T)
        return rely

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
        if "__global__" in x and not self.global_entity:
            x = {k: v for k, v in x.items() if k != "__global__"}
        relative_positions, entity_type = self._relative_positions(
            x, index_map, packpad_index, shape, entity_type
        )
        torientation = (
            self._orientations(x, index_map, packpad_index, shape)
            if self.radial
            else None
        )

        if self.interpolate:
            indices_weights = self._interpolated_partition(
                relative_positions, torientation
            )

            keys: torch.Tensor = sum(  # type: ignore
                self.keys(indices) * weights.unsqueeze(-1)
                for indices, weights in indices_weights
            )
            if self.per_entity_values:
                per_entity_indices_weights = [
                    (
                        indices
                        + (entity_type * self.positions)
                        .transpose(2, 1)
                        .long()
                        .repeat(1, indices.size(2), 1),
                        weights,
                    )
                    for indices, weights in indices_weights
                ]
            else:
                per_entity_indices_weights = indices_weights
            values: torch.Tensor = sum(  # type: ignore
                self.values(indices) * weights.unsqueeze(-1)
                for indices, weights in per_entity_indices_weights
            )
        else:
            indices = self._partition(relative_positions, torientation)
            # Batch x Seq x Seq x d_model
            keys = self.keys(indices)
            if self.per_entity_values:
                per_entity_type_indices = indices + (
                    entity_type * self.positions
                ).transpose(2, 1).long().repeat(1, indices.size(2), 1)
            else:
                per_entity_type_indices = indices
            values = self.values(per_entity_type_indices)

        if self.value_relpos_projection or self.key_relpos_projection:
            # TODO: torch.norm deprecated, does this do the right thing?
            dist = relative_positions.norm(p=2, dim=-1).unsqueeze(-1)
            relpos_dist = torch.cat([relative_positions, dist], dim=-1)
            norm_relpos_dist = self.relpos_norm(relpos_dist)
            if self.per_entity_projections:
                # TODO: efficiency
                if self.value_relpos_projection:
                    for i, vproj in enumerate(self.vproj.values()):  # type: ignore
                        v = vproj(norm_relpos_dist)
                        v[entity_type.squeeze(-1) != 0, :, :] = 0.0
                        values += v
                if self.key_relpos_projection:
                    for i, kproj in enumerate(self.kproj.values()):  # type: ignore
                        k = kproj(norm_relpos_dist)
                        k[entity_type.squeeze(-1) != 0, :, :] = 0.0
                        keys += k
            else:
                if self.value_relpos_projection:
                    values += self.vproj(norm_relpos_dist)
                if self.key_relpos_projection:
                    keys += self.kproj(norm_relpos_dist)

        return keys, values

    def _relative_positions(
        self,
        x: Mapping[str, torch.Tensor],
        index_map: torch.Tensor,
        packpad_index: Optional[torch.Tensor],
        shape: RaggedBufferI64,
        entity_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = []
        for entity_name, features in x.items():
            if entity_name not in self.exclude_entities:
                positions.append(
                    features[:, self.position_feature_indices[entity_name]]
                )
            else:
                # TODO: add padding or something?
                raise NotImplementedError(
                    "exclude_entities not implemented for relative position encoding"
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
            size1 = shape.size1(0) if len(tpos) > 0 else 0
            tpos = tpos.reshape(shape.size0(), size1, 2)
            entity_type = entity_type.reshape(shape.size0(), size1, 1)
        # Batch x Seq(q) x Seq(k) x Pos relative positions
        return tpos.unsqueeze(2) - tpos.unsqueeze(1), entity_type

    def _orientations(
        self,
        x: Mapping[str, torch.Tensor],
        index_map: torch.Tensor,
        packpad_index: Optional[torch.Tensor],
        shape: RaggedBufferI64,
    ) -> torch.Tensor:
        """
        Maps a sequence of relative positions to indices.
        """
        # Get entity orientations
        orientations = []
        if self.orientation_angle_index is not None:
            for entity_name, feature in x.items():
                feature_index = self.orientation_angle_index.get(entity_name)
                if feature_index is None:
                    orientations.append(
                        torch.zeros_like(feature[:, 0], device=index_map.device)
                    )
                else:
                    orientations.append(feature[:, feature_index])
            torientation = torch.cat(orientations, dim=0)
        elif self.orientation_vec_indices is not None:
            for entity_name, feature in x.items():
                feature_indices = self.orientation_vec_indices.get(entity_name)
                if feature_indices is None:
                    orientations.append(
                        torch.zeros_like(feature[:, 0], device=index_map.device)
                    )
                else:
                    orientation_vec = feature[:, feature_indices]
                    orientation = torch.atan2(
                        orientation_vec[:, 1], orientation_vec[:, 0]
                    )
                    orientations.append(orientation)
            torientation = torch.cat(orientations, dim=0)[:, None]
        else:
            raise ValueError("No orientation information")

        torientation = torientation[index_map]
        if packpad_index is not None:
            return torientation[packpad_index].view(
                packpad_index.size(0), packpad_index.size(1), 1
            )
        else:
            size1 = shape.size1(0) if shape.items() > 0 else 0
            return torientation.reshape(shape.size0(), size1, 1)

    def _partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(q) x Seq(k) x Pos relative positions
        torientation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Maps a sequence of relative positions to indices.
        """
        if self.radial and self.distance:
            return self._polar_partition(relative_positions, torientation)
        elif self.radial:
            return self._radial_partition(relative_positions, torientation)
        elif self.distance:
            return self._distance_partition(relative_positions)
        else:
            return self._grid_partition(relative_positions, torientation)

    def _interpolated_partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(q) x Seq(k) x Pos relative positions
        torientation: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Maps a sequence of relative positions to indices.
        """
        if self.radial and self.distance:
            return self._interpolated_polar_partition(relative_positions, torientation)
        if self.radial:
            return self._interpolated_radial_partition(relative_positions, torientation)
        elif self.distance:
            return self._interpolated_distance_partition(relative_positions)
        else:
            raise NotImplementedError("interpolated grid partition not implemented")

    def _grid_partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(k) x Seq(q) x Pos relative positions
        torientation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert torientation is None, "Not implemented for non-radial"
        clamped_positions = torch.max(
            torch.min(
                self.extent_tensor,  # type: ignore
                relative_positions * (1.0 / self.scale),
            ),
            -self.extent_tensor,  # type: ignore
        )
        positive_positions = clamped_positions + self.extent_tensor
        return (positive_positions * self.strides).sum(dim=-1).round().long()

    def _radial_partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(k) x Seq(q) x Pos relative positions
        torientation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        angles = torch.atan2(
            relative_positions[:, :, :, 1], relative_positions[:, :, :, 0]
        )
        # We need to be careful about ensuring that we don't create indices that fall outside of the extent.
        # Specifically, taking the modulo of a small negative number can round up to the modulus:
        # (torch.tensor([-1e-12], dtype=torch.float32) % torch.tensor([2]).long()).long() == torch.tensor([2]).long()
        # We can avoid this by ensuring that `angles` is always positive.
        if torientation is not None:
            angles = angles - torientation + 2 * math.pi
        return (angles / (2 * math.pi) * self.extent[0] % self.extent[0]).long()

    def _interpolated_radial_partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(k) x Seq(q) x Pos relative positions
        torientation: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        angles = torch.atan2(
            relative_positions[:, :, :, 1], relative_positions[:, :, :, 0]
        )
        if torientation is not None:
            angles = angles - torientation + 2 * math.pi
        norm_angles = angles / (2 * math.pi) * self.extent[0] % self.extent[0]
        index1 = norm_angles.long()
        index2 = (index1 + 1) % self.extent[0]
        weight1 = (index2 - norm_angles) % 1
        weight2 = 1.0 - weight1
        return [(index1, weight1), (index2, weight2)]

    def _distance_partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(k) x Seq(q) x Pos relative positions
    ) -> torch.Tensor:
        distances: torch.Tensor = torch.linalg.norm(relative_positions, dim=-1)
        return torch.min(
            (distances * (1.0 / self.scale)).long(),
            # For polar relative positions, distance extent is last element.
            self.extent_tensor[0, 0, 0, -1] - 1,  # type: ignore
        )

    def _interpolated_distance_partition(
        self,
        relative_positions: torch.Tensor,  # Batch x Seq(k) x Seq(q) x Pos relative positions
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        distances: torch.Tensor = torch.linalg.norm(relative_positions, dim=-1) * (
            1.0 / self.scale
        )
        index1 = torch.min(
            distances.long(),
            self.extent_tensor[0, 0, 0, -1] - 1,  # type: ignore
        )
        index2 = torch.min(
            index1 + 1,
            self.extent_tensor[0, 0, 0, -1] - 1,  # type: ignore
        )
        weight1 = index2 - distances
        weight2 = 1.0 - weight1
        return [(index1, weight1), (index2, weight2)]

    def _polar_partition(
        self,
        relative_positions: torch.Tensor,
        torientation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        aindices = self._radial_partition(relative_positions, torientation)
        dindices = self._distance_partition(relative_positions)
        indices = (torch.stack([aindices, dindices], dim=-1) * self.strides.long()).sum(  # type: ignore
            dim=-1
        )
        return indices

    def _interpolated_polar_partition(
        self,
        relative_positions: torch.Tensor,
        torientation: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        (aindex1, aweight1), (aindex2, aweight2) = self._interpolated_radial_partition(
            relative_positions, torientation
        )
        (dindex1, dweight1), (
            dindex2,
            dweight2,
        ) = self._interpolated_distance_partition(relative_positions)
        indices1 = (torch.stack([aindex1, dindex1], dim=-1) * self.strides.long()).sum(dim=-1)  # type: ignore
        weights1 = aweight1 * dweight1
        indices2 = (torch.stack([aindex2, dindex1], dim=-1) * self.strides.long()).sum(dim=-1)  # type: ignore
        weights2 = aweight2 * dweight1
        indices3 = (torch.stack([aindex1, dindex2], dim=-1) * self.strides.long()).sum(dim=-1)  # type: ignore
        weights3 = aweight1 * dweight2
        indices4 = (torch.stack([aindex2, dindex2], dim=-1) * self.strides.long()).sum(dim=-1)  # type: ignore
        weights4 = aweight2 * dweight2
        return [
            (indices1, weights1),
            (indices2, weights2),
            (indices3, weights3),
            (indices4, weights4),
        ]
