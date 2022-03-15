import numpy as np
from typing import List, Dict, Type, Mapping

import vizdoom as vzd

from entity_gym.environment import (
    CategoricalAction,
    CategoricalActionMask,
    Entity,
    Environment,
    CategoricalActionSpace,
    ActionSpace,
    EpisodeStats,
    ObsSpace,
    Observation,
    Action,
)


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
        self._available_buttons = self._doomgame.get_available_buttons()
        assert len(self._available_buttons) > 0, "No available buttons to the agent. Double-check your config files."
        for action in self._available_buttons:
            action = action.name
            if "DELTA" in action:
                self._action_space[action] = CategoricalActionSpace(DELTA_SPEED_STEPS_STRS)
            else:
                self._action_space[action] = CategoricalActionSpace(["off", "on"])
            self._action_mask[action] = CategoricalActionMask(actor_ids=np.array([0]), mask=None)

    def obs_space(self) -> ObsSpace:
        return self._observation_space

    def action_space(self) -> Dict[str, ActionSpace]:
        return self._action_space

    def _build_state(self, state, reward, terminal):
        """Build Entity Gym state from ViZDoom state"""
        game_variable_list = np.array([state.game_variables], dtype=np.float32)
        object_list = np.array(
            [
                (o.position_x, o.position_y, o.position_z, ord(o.name[0])) for o in state.objects
            ],
            dtype=np.float32
        )
        return Observation(
            features={
                "Player": game_variable_list,
                "Objects": object_list,
            },
            ids={"Player": [0]},
            actions=self._action_mask,
            reward=reward,
            done=terminal,
            end_of_episode_info=EpisodeStats(
                length=self._episode_steps, total_reward=self._sum_reward
            )
            if terminal else None,
        )

    def _enn_action_to_doom(self, action):
        doom_action = [0 for _ in range(len(self._available_buttons))]
        for i, button in enumerate(self._available_buttons):
            # There is only one actor (player)
            button_action = action[button.name].actions[0]
            if "DELTA" in button.name:
                doom_action[i] = DELTA_SPEED_STEPS[button_action]
            else:
                doom_action[i] = button_action
        return doom_action

    def act(self, action: Mapping[str, Action]) -> Observation:
        # TODO update
        doom_action = self._enn_action_to_doom(action)
        reward = self._doomgame.make_action(doom_action, self._frame_skip)
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

    def reset(self) -> Observation:
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
