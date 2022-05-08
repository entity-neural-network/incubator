from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import ragged_buffer
from ragged_buffer import RaggedBufferF32

from entity_gym.env import ObsSpace


@dataclass
class TranslationConfig:
    """Settings for translating position features.

    Attributes:
        reference_entity: Entity type of the entity which will be placed at the origin.
        position_features: Names of position features used for translation.
        rotation_vec_features: Names of that gives the direction of the reference entity in radians. All entities are rotated by this value.
        rotation_angle_feature: Name of feature that gives the direction of the reference entity in radians. All entities are rotated by this value.
        add_dist_feature: Adds a feature that is the distance to the reference entity.
    """

    reference_entity: str
    position_features: List[str]
    rotation_vec_features: Optional[List[str]] = None
    rotation_angle_feature: Optional[str] = None
    add_dist_feature: bool = False

    def __post_init__(self) -> None:
        assert (
            self.rotation_vec_features is None or self.rotation_angle_feature is None
        ), "Only one of rotation_vec_features and rotation_angle_feature can be specified"


class TranslatePositions(TranslationConfig):
    def __init__(
        self,
        cfg: TranslationConfig,
        obs_space: ObsSpace,
    ):
        super().__init__(**cfg.__dict__)
        self.feature_indices = {
            entity_name: [
                entity.features.index(feature_name)
                for feature_name in cfg.position_features
            ]
            for entity_name, entity in obs_space.entities.items()
            if entity_name != cfg.reference_entity
            and all(
                [
                    feature_name in entity.features
                    for feature_name in cfg.position_features
                ]
            )
        }
        self.reference_indices = [
            obs_space.entities[cfg.reference_entity].features.index(feature_name)
            for feature_name in cfg.position_features
        ]
        self.orientation_vec_indices = (
            [
                obs_space.entities[cfg.reference_entity].features.index(feature_name)
                for feature_name in cfg.rotation_vec_features
            ]
            if cfg.rotation_vec_features is not None
            else None
        )
        self.orientation_angle_index = (
            obs_space.entities[cfg.reference_entity].features.index(
                cfg.rotation_angle_feature
            )
            if cfg.rotation_angle_feature is not None
            else None
        )
        self.reference_entity = cfg.reference_entity
        self.add_dist_feature = cfg.add_dist_feature

    def apply(self, entities: Dict[str, RaggedBufferF32]) -> None:
        if self.reference_entity not in entities:
            return
        reference_entity = entities[self.reference_entity]
        origin = reference_entity[:, :, self.reference_indices]
        if self.orientation_vec_indices is not None:
            orientation: Optional[RaggedBufferF32] = reference_entity[
                :, :, self.orientation_vec_indices
            ]
        elif self.orientation_angle_index is not None:
            angle = reference_entity[:, :, self.orientation_angle_index].as_array()
            orientation = RaggedBufferF32.from_array(
                np.hstack([np.cos(angle), np.sin(angle)]).reshape(-1, 1, 2)
            )
            # TODO: ragged_buffer.translate_rotate assumes that all input arguments are views, so apply identity view
            orientation = orientation[:, :, :]
        else:
            orientation = None
        for entity_name, indices in self.feature_indices.items():
            if entity_name in entities:
                if orientation is not None:
                    ragged_buffer.translate_rotate(
                        entities[entity_name][:, :, indices],
                        origin,
                        orientation,
                    )
                else:
                    feats = entities[entity_name][:, :, indices]
                    feats -= origin

                if self.add_dist_feature:
                    # TODO: efficiency
                    ea = entities[entity_name].as_array()
                    np.linalg.norm(ea[:, indices], axis=1).reshape(-1, 1)
                    entities[entity_name] = RaggedBufferF32.from_flattened(
                        np.concatenate(
                            [ea, np.linalg.norm(ea[:, indices], axis=1).reshape(-1, 1)],
                            axis=1,
                        ),
                        entities[entity_name].size1(),
                    )

    def transform_obs_space(self, obs_space: ObsSpace) -> ObsSpace:
        if self.add_dist_feature:
            obs_space = deepcopy(obs_space)
            for entity_name in self.feature_indices.keys():
                features = list(obs_space.entities[entity_name].features)
                features.append("TranslatePositions.distance")
                obs_space.entities[entity_name].features = features
        return obs_space
