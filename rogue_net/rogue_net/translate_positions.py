from dataclasses import dataclass
from typing import Dict, List, Mapping
from entity_gym.environment import ObsSpace
from ragged_buffer import RaggedBufferF32
import torch


class TranslatePositions:
    def __init__(
        self, reference_entity: str, position_features: List[str], obs_space: ObsSpace
    ):
        self.feature_indices = {
            entity_name: [
                entity.features.index(feature_name)
                for feature_name in position_features
            ]
            for entity_name, entity in obs_space.entities.items()
            if entity_name != reference_entity
            and all(
                [feature_name in entity.features for feature_name in position_features]
            )
        }
        self.reference_indices = [
            obs_space.entities[reference_entity].features.index(feature_name)
            for feature_name in position_features
        ]
        self.reference_entity = reference_entity

    def apply(self, entities: Dict[str, RaggedBufferF32]) -> None:
        if self.reference_entity not in entities:
            return
        reference_entity = entities[self.reference_entity]
        origin = reference_entity[:, :, self.reference_indices]
        for entity_name, indices in self.feature_indices.items():
            if entity_name in entities:
                feats = entities[entity_name][:, :, indices]
                feats -= origin
