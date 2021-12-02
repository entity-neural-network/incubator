import os
from abc import abstractmethod
from typing import Type, Dict, Optional, Any

from enn_zoo.griddly.wrapper import GriddlyEnv
from entity_gym.environment import ActionSpace, ObsSpace, Entity, CategoricalActionSpace
from entity_gym.environment import Environment
from griddly import GymWrapper

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
    env.close()

    class InstantiatedGriddlyEnv(GriddlyEnv):
        @classmethod
        def _griddly_env(cls) -> Any:
            return GymWrapper(
                yaml_file=yaml_file,
                image_path=image_path,
                shader_path=shader_path,
                level=level,
            )

        @classmethod
        def obs_space(cls) -> ObsSpace:
            return observation_space

        @classmethod
        def action_space(cls) -> Dict[str, ActionSpace]:
            return action_space

    return InstantiatedGriddlyEnv


GRIDDLY_ENVS: Dict[str, Type[Environment]] = {
    "GDY-Clusters-0": create_env(
        os.path.join(init_path, "env_descriptions/clusters.yaml"), level=0
    ),
    "GDY-Clusters-1": create_env(
        os.path.join(init_path, "env_descriptions/clusters.yaml"), level=1
    ),
    "GDY-Clusters-2": create_env(
        os.path.join(init_path, "env_descriptions/clusters.yaml"), level=2
    ),
    "GDY-Clusters-3": create_env(
        os.path.join(init_path, "env_descriptions/clusters.yaml"), level=3
    ),
    "GDY-Clusters-4": create_env(
        os.path.join(init_path, "env_descriptions/clusters.yaml"), level=4
    ),
}
