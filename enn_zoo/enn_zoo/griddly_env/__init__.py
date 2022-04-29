import os
from typing import Any, Dict, Optional, Type, Mapping

import numpy as np
from griddly import GymWrapper, gd

from enn_zoo.griddly_env.level_generators.clusters_generator import (
    ClustersLevelGenerator,
)
from enn_zoo.griddly_env.level_generators.crafter_generator import CrafterLevelGenerator
from enn_zoo.griddly_env.level_generators.level_generator import LevelGenerator
from enn_zoo.griddly_env.wrapper import GriddlyEnv
from entity_gym.environment import (
    ActionSpace,
    CategoricalActionSpace,
    Entity,
    Observation,
    ObsSpace,
    Action,
)

init_path = os.path.dirname(os.path.realpath(__file__))


def generate_obs_space(env: Any) -> ObsSpace:
    # Each entity contains x, y, z positions, plus the values of all variables
    global_variables = env.game.get_global_variable_names()

    # Global entity for global variables and global actions (these dont really exist in Griddly)
    space = {"__global__": Entity(global_variables)}
    for name, features in env.observation_space.features.items():
        space[name] = Entity(features)

    return ObsSpace(space)


def generate_action_space(env: Any) -> Dict[str, ActionSpace]:
    action_space: Dict[str, ActionSpace] = {}

    flat_action_mapping = []

    actions = []
    actions.append("NOP")
    flat_action_mapping.append([0, 0])
    for action_type_id, action_name in enumerate(env.action_names):
        action_mapping = env.action_input_mappings[action_name]
        input_mappings = action_mapping["InputMappings"]

        for action_id in range(1, len(input_mappings) + 1):
            mapping = input_mappings[str(action_id)]
            description = mapping["Description"]
            actions.append(description)

            flat_action_mapping.append([action_type_id, action_id])

        action_space["flat"] = CategoricalActionSpace(actions)

    return action_space, flat_action_mapping


def create_env(
        yaml_file: str,
        global_observer_type: Any = gd.ObserverType.SPRITE_2D,
        image_path: Optional[str] = None,
        shader_path: Optional[str] = None,
        level: int = 0,
        random_levels: bool = False,
        level_generator: Optional[LevelGenerator] = None,
) -> Type[GriddlyEnv]:
    """
    In order to fit the API for the Environment, we need to pre-load the environment from the yaml and then pass in
    observation space, action space and the instantiated GymWrapper
    """

    env = GymWrapper(
        yaml_file=yaml_file,
        player_observer_type=gd.ObserverType.ENTITY,
        image_path=image_path,
        shader_path=shader_path,
        level=level,
    )
    env.reset()
    action_space, flat_action_mapping = generate_action_space(env)
    observation_space = generate_obs_space(env)
    level_count = env.level_count
    env.close()

    class InstantiatedGriddlyEnv(GriddlyEnv):
        @classmethod
        def _griddly_env(cls) -> Any:
            return GymWrapper(
                yaml_file=yaml_file,
                image_path=image_path,
                shader_path=shader_path,
                player_observer_type=gd.ObserverType.ENTITY,
                global_observer_type=global_observer_type,
                level=level,
            )

        @classmethod
        def obs_space(cls) -> ObsSpace:
            return observation_space

        @classmethod
        def action_space(cls) -> Dict[str, ActionSpace]:
            return action_space

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
                return np.array(flat_action_mapping[action['flat'].actions[0]])

        def reset(self) -> Observation:

            self.total_reward = 0
            self.step = 0

            if random_levels:
                random_level = np.random.choice(level_count)
                obs = self._env.reset(level_id=random_level)
                return self._make_observation(obs)
            elif isinstance(level_generator, LevelGenerator):
                level_string = level_generator.generate()
                obs = self._env.reset(level_string=level_string)
                return self._make_observation(obs)
            else:
                obs = self._env.reset()
                return self._make_observation(obs)

    return InstantiatedGriddlyEnv


GRIDDLY_ENVS: Dict[str, Dict[str, Any]] = {
    "GDY-Clusters-Multi-Generated-Small": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 10,
                "height": 10,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 5,
                "m_blue": 5,
                "m_green": 5,
                "m_spike": 5,
                "walls": False,
                "avatar": False,
            }
        ),
    },
    "GDY-Clusters-Multi-Generated-Medium": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 20,
                "height": 20,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 10,
                "m_blue": 10,
                "m_green": 10,
                "m_spike": 10,
                "walls": False,
                "avatar": False,
            }
        ),
    },
    "GDY-Clusters-Multi-Generated-Large": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 50,
                "height": 50,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 20,
                "m_blue": 20,
                "m_green": 20,
                "m_spike": 20,
                "walls": False,
                "avatar": False,
            }
        ),
    },
    "GDY-Clusters-Multi-All": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "random_levels": True,
    },
    "GDY-Clusters-Multi-0": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 0,
    },
    "GDY-Clusters-Multi-1": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 1,
    },
    "GDY-Clusters-Multi-2": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 2,
    },
    "GDY-Clusters-Multi-3": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 3,
    },
    "GDY-Clusters-Multi-4": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 4,
    },
    "GDY-Clusters-Generated-Small": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 10,
                "height": 10,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 5,
                "m_blue": 5,
                "m_green": 5,
                "m_spike": 5,
                "walls": False,
                "avatar": True,
            }
        ),
    },
    "GDY-Clusters-Generated-Medium": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 20,
                "height": 20,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 10,
                "m_blue": 10,
                "m_green": 10,
                "m_spike": 10,
                "walls": False,
                "avatar": True,
            }
        ),
    },
    "GDY-Clusters-Generated-Large": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 50,
                "height": 50,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 20,
                "m_blue": 20,
                "m_green": 20,
                "m_spike": 20,
                "walls": False,
                "avatar": True,
            }
        ),
    },
    "GDY-Clusters-All": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "random_levels": True,
    },
    "GDY-Clusters-0": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 0,
    },
    "GDY-Clusters-1": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 1,
    },
    "GDY-Clusters-2": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 2,
    },
    "GDY-Clusters-3": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 3,
    },
    "GDY-Clusters-4": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 4,
    },
    ############ Grafter Envs ############
    "GDY-Grafter-Single-30": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_single.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 1),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-Single-50": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_single.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 1),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-Single-100": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_single.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 1),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-4Player-30": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_multi_4.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 4),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-4Player-50": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/rafter_multi_4.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 4),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-4Player-100": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_multi_4.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 4),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-8Player-50": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_multi_8.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 8),
        "image_path": os.path.join(init_path, "images/grafter")
    },
    "GDY-Grafter-8Player-100": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_multi_8.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 8),
        "image_path": os.path.join(init_path, "images/grafter")

    },
    "GDY-Grafter-8Player-200": {
        "yaml_file": os.path.join(init_path, "env_descriptions/grafter/grafter_multi_8.yaml"),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 8),
        "image_path": os.path.join(init_path, "images/grafter")
    },

}
