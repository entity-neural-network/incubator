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

    @abstractmethod
    def _to_griddly_action(self, action: Mapping[str, Action]) -> np.ndarray:
        pass

    def _make_observation(
        self, obs: Dict[str, Any], reward: int = 0, done: bool = False
    ) -> Observation:
        entities = obs["Entities"]
        entity_ids = obs["Ids"]
        actor_masks = obs["ActorMasks"]
        actor_ids = obs["ActorIds"]

        self.entity_locations = obs["Locations"]

        global_features = None
        if "__global__" in entities:
            global_features = entities["__global__"][0]
            # entities["__griddly_global__"] = entities["__global__"]
            # entity_ids["__griddly_global__"] = entity_ids["__global__"]
            del entities["__global__"]
            del entity_ids["__global__"]

        entities = {
            name: np.array(obs, dtype=np.float32) for name, obs in entities.items()
        }

        flat_actor_ids = []
        flat_actor_masks = []

        flat_action_accumulate = {}

        for action_name in self._env.action_names:
            if action_name not in actor_ids or action_name not in actor_masks:
                continue
            for action_id, actor_mask in zip(
                actor_ids[action_name], actor_masks[action_name]
            ):
                if action_id not in flat_action_accumulate:
                    flat_action_accumulate[action_id] = [1]

                flat_action_accumulate[action_id].extend(actor_mask[1:])

        for actor_id, actor_mask in flat_action_accumulate.items():
            flat_actor_ids.append(actor_id)
            flat_actor_masks.append(actor_mask)

        action_masks = {
            "flat": CategoricalActionMask(
                actor_ids=flat_actor_ids,
                mask=np.array(flat_actor_masks).astype(np.bool_),
            )
        }

        return Observation(
            global_features=global_features,
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
