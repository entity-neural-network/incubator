from typing import Mapping, Dict

import numpy as np
from entity_gym.environment import Environment, Action, Observation, ActionSpace, ObsSpace

from griddly import GymWrapper


class ENNWrapper(Environment):
    """
    Griddly Environments are defined in yaml files.

    This wrapper loads the environment and extracts what it needs from the environment description to make any
    Griddly Environment compatible with the Entity Neural Network API
    """

    def __init__(self, yaml_file: str):
        self._env = GymWrapper(yaml_file=yaml_file)

        self._obs_space = _generate_obs_space()
        self._action_space = _generate_action_space()

        self._entity_names = eng.game.get_object_names()

    def _generate_obs_space(self) -> ObsSpace:
        variable_names = self._env.game.get_variable_names()

        # TODO: currently we flatten out all possible variables regardless of entity.
        space = {}
        for name in self._entity_names:
            space[name] = Entity(['x', 'y', 'z', *variable_names])

        return ObsSpace(space)

    def _generate_action_space(self) -> Dict[str, ActionSpace]:
        # TODO: currently we flatten out all possible actions regardless of entity.
        # This circumvents https://github.com/entity-neural-network/incubator/issues/19
        return self._action_space

    def _get_entity_observation(self) -> Dict[str, np.ndarray]:
        g_state = env.get_state()

    def _get_action_masks(self) -> Mapping[str, ActionMask]:

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return self._obs_space

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return self._action_space

    def _reset(self) -> Observation:
        self._env.reset()

        entities = self._get_entity_observation()
        action_masks = self._get_action_masks()

        return Observation(
            entities=entities,
            ids=ids,
            action_masks=action_masks,
            reward=0,
            done=False
        )

    def _act(self, action: Mapping[str, Action]) -> Observation:
        pass
