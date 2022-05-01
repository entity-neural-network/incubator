from abc import abstractmethod
from typing import Any, Dict, Mapping

import numpy as np
import numpy.typing as npt

from entity_gym.environment import (
    Action,
    ActionSpace,
    CategoricalActionMask,
    Environment,
    Observation,
    ObsSpace,
)
from entity_gym.environment.environment import CategoricalAction


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

        if len(self._env.action_space_parts) > 2:
            entity_actions = []
            for action_name, a in action.items():
                action_type = self._env.action_names.index(action_name)
                for entity_id, action_id in a.items():
                    entity_location = self.entity_locations[entity_id]
                    entity_actions.append(
                        np.array(
                            [
                                entity_location[0],
                                entity_location[1],
                                action_type,
                                action_id,
                            ]
                        )
                    )

            return np.stack(entity_actions)
        else:

            for action_name, a in action.items():
                action_type = self._env.action_names.index(action_name)
                assert isinstance(a, CategoricalAction)
                # TODO: this only works if we have a single entity, otherwise we have to map the entityID to an x,y coordinate
                action_id = a.actions[0]

            return np.array([action_type, action_id])

    def _make_observation(
        self, obs: Dict[str, Any], reward: int = 0, done: bool = False
    ) -> Observation:
        entities = obs["Entities"]
        entity_ids = obs["Ids"]
        actor_masks = obs["ActorMasks"]
        actor_ids = obs["ActorIds"]

        self.entity_locations = obs["Locations"]

        entities = {
            name: np.array(obs, dtype=np.float32) for name, obs in entities.items()
        }

        action_masks = {}
        for action_name, actor_mask in actor_masks.items():
            action_masks[action_name] = CategoricalActionMask(
                actor_ids=actor_ids[action_name],
                mask=np.array(actor_mask).astype(np.bool_),
            )

        return Observation(
            features=entities,
            ids=entity_ids,
            actions=action_masks,
            reward=reward,
            done=done,
        )

    def reset(self) -> Observation:
        obs = self._env.reset()

        self.total_reward = 0
        self.step = 0

        return self._make_observation(obs)

    def act(self, actions: Mapping[str, Action]) -> Observation:
        g_action = self._to_griddly_action(actions)
        obs, reward, done, info = self._env.step(g_action)

        self.total_reward += reward
        self.step += 1

        return self._make_observation(obs, reward, done)

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return self._env.render(**kwargs, observer="global")  # type: ignore
