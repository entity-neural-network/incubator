import os
from typing import Dict, Type

from entity_gym.environment import (
    ActionSpace,
    ObsSpace,
)
from enn_zoo.vizdoom.vizdoom_environment import DoomEntityEnvironment


init_path = os.path.dirname(os.path.realpath(__file__))

VIZDOOM_ENVS = {
    "DoomHealthGathering": os.path.join(init_path, "scenarios/health_gathering.cfg"),
    "DoomHealthGatheringSupreme": os.path.join(init_path, "scenarios/health_gathering_supreme.cfg"),
}


def create_vizdoom_env(
    config_file_path: str,
) -> Type[DoomEntityEnvironment]:
    """
    In order to fit the API for the Environment, we need to pre-load the environment from the config file then pass in
    observation space, action space.
    Copied from the Griddly code.
    """

    env = DoomEntityEnvironment(config_file_path)
    observation_space = env._observation_space
    action_space = env._action_space

    class InstantiatedDoomEntityEnvironment(DoomEntityEnvironment):
        def __init__(self, frame_skip: int = 4):
            super().__init__(config_file_path, frame_skip)

        @classmethod
        def obs_space(cls) -> ObsSpace:
            return observation_space

        @classmethod
        def action_space(cls) -> Dict[str, ActionSpace]:
            return action_space

    return InstantiatedDoomEntityEnvironment
