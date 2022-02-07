import os
import types

import numpy as np
from typing import Tuple, Type, Dict, Optional, Any, Callable

from enn_zoo.griddly.level_generators.clusters_generator import ClustersLevelGenerator
from enn_zoo.griddly.level_generators.level_generator import LevelGenerator
from enn_zoo.griddly.wrapper import GriddlyEnv
from entity_gym.environment import (
    ActionSpace,
    ObsSpace,
    Entity,
    CategoricalActionSpace,
    Observation,
)
from griddly import GymWrapper, gd

init_path = os.path.dirname(os.path.realpath(__file__))


def generate_obs_space(env: Any) -> ObsSpace:
    # Each entity contains x, y, z positions, plus the values of all variables
    global_variables = env.game.get_global_variable_names()
    object_variable_map = env.game.get_object_variable_map()

    # Global entity for global variables and global actions (these dont really exist in Griddly)
    space = {"__global__": Entity(global_variables)}
    for name in env.object_names:
        space[name] = Entity(
            ["x", "y", "z", "orientation", "player_id", *object_variable_map[name]]
        )

    return ObsSpace(space)


def generate_action_space(env: Any) -> Dict[str, ActionSpace]:
    action_space: Dict[str, ActionSpace] = {}
    for action_name, action_mapping in env.action_input_mappings.items():
        # Ignore internal actions for the action space
        if action_mapping["Internal"] == True:
            continue

        input_mappings = action_mapping["InputMappings"]

        actions = []
        actions.append("NOP")  # In Griddly, Action ID 0 is always NOP
        for action_id in range(1, len(input_mappings) + 1):
            mapping = input_mappings[str(action_id)]
            description = mapping["Description"]
            actions.append(description)

        action_space[action_name] = CategoricalActionSpace(actions)

    return action_space


def create_env(
    yaml_file: str,
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
        yaml_file=yaml_file, image_path=image_path, shader_path=shader_path, level=level
    )
    env.reset()
    action_space = generate_action_space(env)
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
                player_observer_type=gd.ObserverType.NONE,
                global_observer_type=gd.ObserverType.BLOCK_2D,
                level=level,
            )

        @classmethod
        def obs_space(cls) -> ObsSpace:
            return observation_space

        @classmethod
        def action_space(cls) -> Dict[str, ActionSpace]:
            return action_space

        def reset(self) -> Observation:
            if random_levels:
                random_level = np.random.choice(level_count)
                self._env.reset(level_id=random_level)
            elif isinstance(level_generator, LevelGenerator):
                level_string = level_generator.generate()
                self._env.reset(level_string=level_string)
            else:
                self._env.reset()

            self.total_reward = 0
            self.step = 0

            return self._make_observation()

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
}
