from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt
from griddly import GymWrapper

from enn_zoo.griddly_env.level_generators.level_generator import LevelGenerator
from entity_gym.env import (
    Action,
    ActionSpace,
    CategoricalActionMask,
    CategoricalActionSpace,
    Entity,
    Environment,
    Observation,
    ObsSpace,
)
from entity_gym.env.environment import CategoricalAction


class GriddlyEnv(Environment):
    def __init__(
        self,
        random_levels: bool = False,
        level_generator: Optional[LevelGenerator] = None,
        **kwargs: Any
    ) -> None:

        self._env = GymWrapper(**kwargs)

        self._env.reset()

        self._observation_space = self._generate_obs_space()
        self._action_space, self._flat_action_mapping = self._generate_action_space()
        self._random_levels = random_levels
        self._level_generator = level_generator

    def obs_space(self) -> ObsSpace:
        return self._observation_space

    def action_space(self) -> Dict[str, ActionSpace]:
        return self._action_space

    def _generate_obs_space(self) -> ObsSpace:
        # Each entity contains x, y, z positions, plus the values of all variables
        global_features: List[str] = []
        if "__global__" in self._env.observation_space.features:
            global_features = self._env.observation_space.features["__global__"]
            del self._env.observation_space.features["__global__"]

        space = {
            name: Entity(features)
            for name, features in self._env.observation_space.features.items()
        }
        return ObsSpace(global_features=global_features, entities=space)

    def _generate_action_space(self) -> Tuple[Dict[str, ActionSpace], List[List[int]]]:
        action_space: Dict[str, ActionSpace] = {}
        flat_action_mapping: List[List[int]] = []

        if len(self._env.action_space_parts) > 2:
            for action_type_id, action_name in enumerate(self._env.action_names):
                action_mapping = self._env.action_input_mappings[action_name]
                input_mappings = action_mapping["InputMappings"]

                actions = []
                actions.append("NOP")  # In Griddly, Action ID 0 is always NOP
                for action_id in range(1, len(input_mappings) + 1):
                    mapping = input_mappings[str(action_id)]
                    description = mapping["Description"]
                    actions.append(description)

                action_space[action_name] = CategoricalActionSpace(actions)
        else:
            actions = []
            actions.append("NOP")
            flat_action_mapping.append([0, 0])
            for action_type_id, action_name in enumerate(self._env.action_names):
                action_mapping = self._env.action_input_mappings[action_name]
                input_mappings = action_mapping["InputMappings"]

                for action_id in range(1, len(input_mappings) + 1):
                    mapping = input_mappings[str(action_id)]
                    description = mapping["Description"]
                    actions.append(description)

                    flat_action_mapping.append([action_type_id, action_id])

            action_space["flat"] = CategoricalActionSpace(actions)

        return action_space, flat_action_mapping

    def _to_griddly_action(self, action: Mapping[str, Action]) -> np.ndarray:
        if len(self._env.action_space_parts) > 2:
            entity_actions = []
            for action_name, a in action.items():
                action_type = self._env.action_names.index(action_name)
                assert isinstance(a, CategoricalAction)
                for entity_id, action_id in zip(a.actors, a.indices):
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
            single_action = action["flat"]
            assert isinstance(single_action, CategoricalAction)
            return np.array(self._flat_action_mapping[single_action.indices[0]])

    def make_observation(
        self,
        obs: Dict[str, Any],
        reward: int = 0,
        done: bool = False,
    ) -> Observation:
        entities = obs["Entities"]
        entity_ids = obs["Ids"]
        actor_masks = obs["ActorMasks"]
        actor_ids = obs["ActorIds"]

        self.entity_locations = obs["Locations"]

        global_features = None
        if "__global__" in entities:
            global_features = entities["__global__"][0]
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
        self.total_reward = 0
        self.step = 0

        if self._random_levels:
            random_level = np.random.choice(self._env.level_count)
            obs = self._env.reset(level_id=random_level)
            return self.make_observation(obs)
        elif isinstance(self._level_generator, LevelGenerator):
            level_string = self._level_generator.generate()
            obs = self._env.reset(level_string=level_string)
            return self.make_observation(obs)
        else:
            obs = self._env.reset()
            return self.make_observation(obs)

    def act(self, actions: Mapping[str, Action]) -> Observation:
        g_action = self._to_griddly_action(actions)
        obs, reward, done, info = self._env.step(g_action)

        self.total_reward += reward
        self.step += 1

        return self.make_observation(obs, reward, done)

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return self._env.render(**kwargs, observer="global")  # type: ignore
