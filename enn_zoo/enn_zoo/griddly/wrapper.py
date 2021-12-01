from abc import abstractmethod
from collections import defaultdict
from typing import Mapping, Dict, Set, Tuple, Any

import numpy as np
from entity_gym.environment import (
    Environment,
    Action,
    Observation,
    ActionSpace,
    DenseCategoricalActionMask,
    ObsSpace,
    ActionMask,
)
from griddly import GymWrapper


class GriddlyEnv(Environment):
    def __init__(self) -> None:

        # Create an instantiation of the griddly envs
        self._env = self.__class__._griddly_env()

        self._env.reset()

        self._obs_space = self.__class__.obs_space()
        self._action_space = self.__class__.action_space()

        self._current_g_state = self._env.get_state()

    @classmethod
    @abstractmethod
    def _griddly_env(cls) -> Any:
        pass

    @classmethod
    @abstractmethod
    def obs_space(cls) -> ObsSpace:
        pass

    @classmethod
    @abstractmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        pass

    def _to_griddly_action(self, action: Mapping[str, Action]) -> np.ndarray:

        for action_name, a in action.items():
            action_type = self._env.action_names.index(action_name)
            # TODO: this only works if we have a single entity, otherwise we have to map the entityID to an x,y coordinate
            action_id = a.actions[0][1]

        return np.array([action_type, action_id])

    def _get_entity_observation(self) -> Tuple[Set[Any], Dict[str, np.ndarray]]:
        self._current_g_state = self._env.get_state()

        def orientation_feature(orientation_string: str) -> int:
            if orientation_string == "NONE":
                return 0
            elif orientation_string == "UP":
                return 0
            elif orientation_string == "RIGHT":
                return 1
            elif orientation_string == "DOWN":
                return 2
            elif orientation_string == "LEFT":
                return 3
            else:
                raise RuntimeError("Unknown Orientation")

        # TODO: Push this down into a c++ helper to make it speedy
        entity_observation = defaultdict(list)
        entity_ids = set()
        for i, object in enumerate(self._current_g_state["Objects"]):
            name = object["Name"]
            location = object["Location"]
            variables = object["Variables"]

            # entity_ids.add(f'{location[0]},{location[1]}')
            # TODO: currently entity ids are a bit meaningless, but they have to be int or things break deeper down
            entity_ids.add(i)

            feature_vec = np.zeros(
                len(self._obs_space.entities[name].features), dtype=np.float32
            )
            feature_vec[0] = location[0]
            feature_vec[1] = location[1]
            feature_vec[2] = 0
            feature_vec[3] = orientation_feature(object["Orientation"])
            feature_vec[4] = object["PlayerId"]
            for i, variable_name in enumerate(self._env.variable_names):
                feature_vec[5 + i] = (
                    variables[variable_name] if variable_name in variables else 0
                )

            entity_observation[name].append(feature_vec)

        return (
            entity_ids,
            {name: np.stack(features) for name, features in entity_observation.items()},
        )

    def _get_action_masks(self) -> Mapping[str, DenseCategoricalActionMask]:

        # TODO: Push this down into c++ helper?
        # TODO: currently hard coded to only get action masks for player 1
        # TODO: assuming we only have a single actor entity 'avatar'
        mask_for_action = {}
        entity_id_for_action = {}
        for action_name in self._env.action_names:
            mask_for_action[action_name] = np.zeros(
                len(self._action_space[action_name].choices)  # type: ignore
            )

            # TODO: single actor identity is hard coded to 0, when this is fixed,
            #  we assign the actions to their possible actors
            entity_id_for_action[action_name] = 0

        for location, available_action_types in self._env.game.get_available_actions(
            1
        ).items():
            available_action_ids = self._env.game.get_available_action_ids(
                location, list(available_action_types)
            )
            for action_name, action_ids in available_action_ids.items():
                mask_for_action[action_name][action_ids] = 1

        action_mask_mapping = {}
        for action_name in mask_for_action.keys():
            mask = mask_for_action[action_name]
            entity_id = entity_id_for_action[action_name]
            action_mask_mapping[action_name] = DenseCategoricalActionMask(
                actors=np.array([entity_id]), mask=mask.reshape(1, -1)
            )

        return action_mask_mapping

    def _make_observation(self, reward: int = 0, done: bool = False) -> Observation:
        entity_ids, entities = self._get_entity_observation()
        action_masks = self._get_action_masks() if not done else {}

        return Observation(
            entities=entities,
            ids=list(entity_ids),
            action_masks=action_masks,
            reward=reward,
            done=done,
        )

    def _reset(self) -> Observation:
        self._env.reset()

        return self._make_observation()

    def _act(self, action: Mapping[str, Action]) -> Observation:
        g_action = self._to_griddly_action(action)
        _, reward, done, info = self._env.step(g_action)

        return self._make_observation(reward, done)
