from abc import abstractmethod
from typing import Mapping, Dict, Any

import numpy as np
from entity_gym.environment import (
    Environment,
    Action,
    Observation,
    ActionSpace,
    EpisodeStats,
    DenseCategoricalActionMask,
    ObsSpace,
)

import numpy.typing as npt


class GriddlyEnv(Environment):
    def __init__(self) -> None:

        # Create an instantiation of the griddly envs
        self._env = self.__class__._griddly_env()

        self._env.reset()

        self._obs_space = self.__class__.obs_space()
        self._action_space = self.__class__.action_space()

        self._current_g_state = self._env.get_state()

        self._entity_observer = self._env.game.get_entity_observer()

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

    def _make_observation(self, reward: int = 0, done: bool = False) -> Observation:
        griddly_entity_observation = self._entity_observer.observe(1)
        entities = griddly_entity_observation["Entities"]
        entity_ids = griddly_entity_observation["EntityIds"]
        entity_masks = griddly_entity_observation["EntityMasks"]

        entities = {
            name: np.array(obs, dtype=np.float32) for name, obs in entities.items()
        }

        action_masks = {}
        for action_name, entity_mask in entity_masks.items():
            action_masks[action_name] = DenseCategoricalActionMask(
                actors=np.array(entity_mask["ActorIdx"]),
                mask=np.array(entity_mask["Masks"]).astype(np.bool_),
            )

        return Observation(
            entities=entities,
            ids=entity_ids,
            action_masks=action_masks,
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, self.total_reward)
            if done
            else None,
        )

    def _reset(self) -> Observation:
        self._env.reset()

        self.total_reward = 0
        self.step = 0

        return self._make_observation()

    def _act(self, action: Mapping[str, Action]) -> Observation:
        g_action = self._to_griddly_action(action)
        _, reward, done, info = self._env.step(g_action)

        self.total_reward += reward
        self.step += 1

        return self._make_observation(reward, done)

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return self._env.render(**kwargs, observer="global")  # type: ignore
