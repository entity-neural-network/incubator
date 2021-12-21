import os
import numpy as np
from typing import List, Dict, Type

import vizdoom as vzd

from entity_gym.environment import (
    CategoricalAction,
    DenseCategoricalActionMask,
    Entity,
    Environment,
    CategoricalActionSpace,
    ActionSpace,
    EpisodeStats,
    ObsSpace,
    Observation,
    Action,
)

init_path = os.path.dirname(os.path.realpath(__file__))

VIZDOOM_ENVS = {
    "DoomHealthGathering": os.path.join(init_path, "scenarios/health_gathering.cfg"),
    "DoomHealthGatheringSupreme": os.path.join(init_path, "scenarios/health_gathering_supreme.cfg"),
}


# Replace Delta buttons with these discrete steps
DELTA_SPEED_STEPS: List = [-45, -30, -15, -5, -1, 0, 1, 5, 15, 30, 45]
DELTA_SPEED_STEPS_STRS: List = [str(x) for x in DELTA_SPEED_STEPS]

FORCED_GAME_VARIABLES: List = [
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.POSITION_Z,
    vzd.GameVariable.ANGLE,
    vzd.GameVariable.PITCH,
]

# Map GameVariables to functions that take in
# said GameVariable and return something more
# convenient for networks (one-hots, in range [0,1],
# etc)
# TODO removed for now (entity gym has normalization, I guess?)
# GAME_VARIABLE_PROCESSOR = {
#    vzd.GameVariable.HEALTH: lambda health: min(health, 200) / 100,
#    vzd.GameVariable.ARMOR: lambda armor: min(armor, 200) / 200,
#    vzd.GameVariable.SELECTED_WEAPON_AMMO: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.SELECTED_WEAPON: lambda weapon: weapon,
#    vzd.GameVariable.AMMO0: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO1: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO2: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO3: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO4: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO5: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO6: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO7: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO8: lambda ammo: int(ammo > 0),
#    vzd.GameVariable.AMMO9: lambda ammo: int(ammo > 0),
# }


class DoomEntityEnvironment(Environment):
    """
    Wrap ViZDoom environments in an appropiate way for Entity Gym, where instead of images
    the environment returns lists of enemies and walls and sectors and whatnot.
    """
    def __init__(
        self,
        config: str,
        frame_skip: int = 4,
    ):
        super().__init__()
        self._config = config
        self._frame_skip = frame_skip

        # Create game
        self._doomgame = vzd.DoomGame()
        self._doomgame.load_config(self._config)
        self._init_done = False

        # We do not need screen stuff so minimize things
        self._doomgame.set_window_visible(False)
        self._doomgame.set_screen_format(vzd.ScreenFormat.GRAY8)
        self._doomgame.set_screen_resolution(vzd.ScreenResolution.RES_160X120)

        # But we need list of objects
        self._doomgame.set_objects_info_enabled(True)

        # Handle GameVariables. Add the ones we force
        self._game_variables = self._doomgame.get_available_game_variables()
        self._game_variables = list(set(self._game_variables).union(set(FORCED_GAME_VARIABLES)))
        self._doomgame.set_available_game_variables(self._game_variables)
        self._game_variable_names = [game_variable.name for game_variable in self._game_variables]

        # When episode terminates the buffers may be empty, so instead we return the last valid observation
        self._last_observation = None
        self._episode_steps = 0
        self._sum_reward = 0

        self._observation_space = ObsSpace(
            {
                "Player": Entity(self._game_variable_names),
                "Objects": Entity(
                    # TODO "type" here will be just the ord(name[0])
                    ["x_pos", "y_pos", "z_pos", "type"]
                )
                # TODO sectors/lines/walls
            }
        )

        self._action_space = {}
        # You can always execute all actions in all steps, so we create a static mask thing
        self._action_mask = {}
        for action in self._doomgame.get_available_buttons():
            action = action.name
            if "DELTA" in action:
                self._action_space[action] = CategoricalActionSpace(DELTA_SPEED_STEPS_STRS)
            else:
                self._action_space[action] = CategoricalActionSpace(["off", "on"])
            self._action_mask[action] = DenseCategoricalActionMask(actor=np.array([0]), mask=None)

    def obs_space(self) -> ObsSpace:
        return self._observation_space

    def action_space(self) -> Dict[str, ActionSpace]:
        return self._action_space

    def _build_state(self, state, reward, terminal):
        """Build Entity Gym state from ViZDoom state"""
        game_variable_list = np.array(state.game_variables, dtype=np.float32)
        object_list = np.array(
            [
                (o.position_x, o.position_y, o.position_z, ord(o.name[0])) for o in state.objects
            ],
            dtype=np.float32
        )
        return Observation(
            entities={
                "Player": game_variable_list,
                "Objects": object_list,
            },
            ids=[0],
            action_masks=self._action_mask,
            reward=reward,
            done=terminal,
            end_of_episode_info=EpisodeStats(
                length=self._episode_steps, total_reward=self._sum_reward
            )
            if terminal else None,
        )

    def _act(self, action):
        # TODO update
        action = self.action_handler(action)
        reward = self._doomgame.make_action(action, self._frame_skip)
        terminal = self._doomgame.is_episode_finished()
        self._episode_steps += 1
        self._sum_reward += reward
        observation = None
        if terminal:
            # No observation available,
            # give the previous observation
            observation = self._last_observation
            observation.done = True
            observation.end_of_episode_info = EpisodeStats(
                length=self._episode_steps, total_reward=self._sum_reward
            )
        else:
            state = self._doomgame.get_state()
            observation = self._build_state(state, reward, terminal)
        # Keep track of the last_observation
        # in case we hit end of the episode
        # (no state available, give last_observation instead)
        self._last_observation = observation
        return observation

    def initialize(self):
        """
        Initialize the game
        """
        self._init_done = True
        self._doomgame.init()

    def _reset(self):
        if not self._init_done:
            self.initialize()
        self._doomgame.new_episode()
        state = self._doomgame.get_state()
        observation = self._build_state(state, reward=0.0, terminal=False)
        self._last_observation = observation
        self._episode_steps = 0
        self._sum_reward = 0
        return observation

    def __del__(self):
        self._doomgame.close()


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
