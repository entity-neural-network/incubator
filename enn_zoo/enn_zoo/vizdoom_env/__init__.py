import os
from typing import Type

from enn_zoo.vizdoom_env.vizdoom_environment import DoomEntityEnvironment

init_path = os.path.dirname(os.path.realpath(__file__))
VIZDOOM_ENVS = {
    "DoomHealthGathering": os.path.join(init_path, "scenarios/health_gathering.cfg"),
    "DoomHealthGatheringSupreme": os.path.join(
        init_path, "scenarios/health_gathering_supreme.cfg"
    ),
    "DoomDefendTheCenter": os.path.join(init_path, "scenarios/defend_the_center.cfg"),
    "DoomBasic": os.path.join(init_path, "scenarios/basic.cfg"),
}


def create_vizdoom_env(
    config_file_path: str,
) -> Type[DoomEntityEnvironment]:
    """
    In order to fit the API for the Environment, we need to pre-load the environment from the config file then pass in
    observation space, action space.
    Copied from the Griddly code.
    """

    class InstantiatedDoomEntityEnvironment(DoomEntityEnvironment):
        def __init__(self, frame_skip: int = 4):
            super().__init__(config_file_path, frame_skip)

    return InstantiatedDoomEntityEnvironment
