from typing import Dict, List, Optional
from entity_gym.environment import ObsSpace
import ragged_buffer
from ragged_buffer import RaggedBufferF32
import numpy as np
from copy import deepcopy


class TranslatePositions:
    def __init__(
        self,
        reference_entity: str,
        position_features: List[str],
        obs_space: ObsSpace,
        # x and y component of unit vector that give the direction of the reference entity. All other entities are rotated by this vector.
        orientation_features: Optional[List[str]] = None,
        # adds a feature that is the distance to the reference entity
        add_dist_feature: bool = False,
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
        self.reference_orientation_indices = (
            [
                obs_space.entities[reference_entity].features.index(feature_name)
                for feature_name in orientation_features
            ]
            if orientation_features is not None
            else None
        )
        self.reference_entity = reference_entity
        self.add_dist_feature = add_dist_feature

    def apply(self, entities: Dict[str, RaggedBufferF32]) -> None:
        if self.reference_entity not in entities:
            return
        reference_entity = entities[self.reference_entity]
        origin = reference_entity[:, :, self.reference_indices]
        orientation = (
            reference_entity[:, :, self.reference_orientation_indices]
            if self.reference_orientation_indices is not None
            else None
        )
        for entity_name, indices in self.feature_indices.items():
            if entity_name in entities:
                if orientation is not None:
                    ragged_buffer.translate_rotate(
                        entities[entity_name][:, :, indices], origin, orientation
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
                obs_space.entities[entity_name].features.append(
                    "TranslatePositions.distance"
                )
        return obs_space
