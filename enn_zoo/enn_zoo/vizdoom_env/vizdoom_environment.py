from typing import Any, Dict, List, Mapping, Set

import numpy as np
import numpy.typing as npt
import vizdoom as vzd  # type: ignore
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

# Replace Delta buttons with these discrete steps
DELTA_SPEED_STEPS: List = [-45, -30, -15, -5, -1, 0, 1, 5, 15, 30, 45]
DELTA_SPEED_STEPS_STRS: List = [str(x) for x in DELTA_SPEED_STEPS]

# Add player location as part of the observations
FORCED_GAME_VARIABLES: Set = {
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.POSITION_Z,
    vzd.GameVariable.ANGLE,
    vzd.GameVariable.PITCH,
}

RENAME_GAME_VARIABLES: Dict = {
    vzd.GameVariable.POSITION_X.name: "x",
    vzd.GameVariable.POSITION_Y.name: "y",
    vzd.GameVariable.POSITION_Z.name: "z",
    vzd.GameVariable.ANGLE.name: "angle",
    vzd.GameVariable.PITCH.name: "pitch",
}

DEG2RAD_MUL = (2 * np.pi) / 360

# Only give information on
# N nearest objects/wall-lines at most
# (to speed things up)
MAX_OBJECTS = 20


def select_n_closest(
    objects: np.ndarray, coordinates: np.ndarray, n: int = MAX_OBJECTS
) -> Any:
    """
    Select n closest objects to coordinates [shape (2,)] and return new array.
    Assumes two first columns of objects are xy coordinates
    """
    distances = np.sum((objects[:, :2] - coordinates) ** 2, axis=1)
    idxs = np.argpartition(distances, n)
    return objects[idxs[:n]]


class DoomEntityEnvironment(Environment):
    """
    Wrap ViZDoom environments in an appropriate way for Entity Gym, where instead of images
    the environment returns lists of enemies and walls and sectors and whatnot.

    Note that this may not be the most sensible thing to do, but is done as a curious
    experiment for the entity-gym code :).

    The environment will consist of "Player" entity, containing player location and rotation
    info, and then for each entity in the labels buffer we give the location info + the first
    letter of the "type" information (e.g. name), as a very rudimentary way of separating
    objects from each other. 20 closest objects to the player are included.

    If "sectors_info_enabled = true" is defined in the config file, this environment
    will also include information on the nearest walls (but this will slow down training).
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

        # We do not need screen stuff so minimize things (apart from rendering)
        self._doomgame.set_window_visible(False)
        self._doomgame.set_screen_format(vzd.ScreenFormat.CRCGCB)
        self._doomgame.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self._image_width = 160
        self._image_height = 120

        # But we need list of objects
        self._doomgame.set_objects_info_enabled(True)

        # Handle GameVariables. Add the ones we force
        self._game_variables = self._doomgame.get_available_game_variables()
        self._game_variables = list(
            set(self._game_variables).union(set(FORCED_GAME_VARIABLES))
        )
        self._doomgame.set_available_game_variables(self._game_variables)
        self._game_variable_names = [
            game_variable.name for game_variable in self._game_variables
        ]
        # Rename variables where needed
        for original_name, new_name in RENAME_GAME_VARIABLES.items():
            if original_name in self._game_variable_names:
                self._game_variable_names[
                    self._game_variable_names.index(original_name)
                ] = new_name
        self._player_x_index = self._game_variable_names.index("x")
        self._player_y_index = self._game_variable_names.index("y")

        # Find where angle and pitch are so we can turn them into radians
        self._angle_index = self._game_variable_names.index("angle")
        self._pitch_index = self._game_variable_names.index("pitch")

        # When episode terminates the buffers may be empty, so instead we return the last valid observation
        self._last_observation: Observation = None  # type: ignore
        self._last_state = None

        obs_space_dict = {
            "Player": Entity(self._game_variable_names),
            "Objects": Entity(
                # TODO "type" here will be just the ord(name[0])
                ["x", "y", "z", "type"]
            ),
        }

        # If sector buffer is enabled, include wall lines in the list of entities
        self._is_sector_buffer_enabled = self._doomgame.is_sectors_info_enabled()
        if self._is_sector_buffer_enabled:
            obs_space_dict["Walls"] = Entity(["x1", "y1", "x2", "y2", "is_blocking"])

        self._observation_space = ObsSpace(entities=obs_space_dict)

        self._action_space: Dict[str, ActionSpace] = {}
        # You can always execute all actions in all steps, so we create a static mask thing
        self._action_mask = {}
        self._available_buttons = self._doomgame.get_available_buttons()
        assert (
            len(self._available_buttons) > 0
        ), "No available buttons to the agent. Double-check your config files."
        for action in self._available_buttons:
            action = action.name
            if "DELTA" in action:
                self._action_space[action] = CategoricalActionSpace(
                    DELTA_SPEED_STEPS_STRS
                )
            else:
                self._action_space[action] = CategoricalActionSpace(["off", "on"])
            self._action_mask[action] = CategoricalActionMask(actor_ids=[0], mask=None)

    def obs_space(self) -> ObsSpace:
        return self._observation_space

    def action_space(self) -> Dict[str, ActionSpace]:
        return self._action_space

    def _build_state(self, state: Any, reward: float, terminal: bool) -> Observation:
        """Build Entity Gym state from ViZDoom state"""
        game_variable_list = np.array([state.game_variables], dtype=np.float32)
        # Turn degree-angles into radians
        game_variable_list[0, self._angle_index] *= DEG2RAD_MUL
        game_variable_list[0, self._pitch_index] *= DEG2RAD_MUL
        object_list = np.array(
            [
                (o.position_x, o.position_y, o.position_z, ord(o.name[0]))
                for o in state.objects
            ],
            dtype=np.float32,
        )

        if len(object_list) > MAX_OBJECTS:
            player_coordinates = game_variable_list[
                0, [self._player_x_index, self._player_y_index]
            ]
            object_list = select_n_closest(object_list, player_coordinates)

        features = {
            "Player": game_variable_list,
            "Objects": object_list,
        }

        if self._is_sector_buffer_enabled:
            # Include wall lines.
            wall_line_list = []
            for sector in state.sectors:
                for line in sector.lines:
                    wall_line_list.append(
                        (line.x1, line.y1, line.x2, line.y2, float(line.is_blocking))
                    )
            wall_line_array = np.array(wall_line_list, dtype=np.float32)

            if len(wall_line_list) > MAX_OBJECTS:
                player_coordinates = game_variable_list[
                    0, [self._player_x_index, self._player_y_index]
                ]
                wall_line_array = select_n_closest(wall_line_array, player_coordinates)

            # We do our own translation here (player is at origin)
            # as we have two coordinates per line to translate
            wall_line_array[:, [0, 2]] -= game_variable_list[0, self._player_x_index]
            wall_line_array[:, [1, 3]] -= game_variable_list[0, self._player_y_index]

            features["Walls"] = wall_line_array

        return Observation(
            features=features,
            ids={"Player": [0]},
            actions=self._action_mask,
            reward=reward,
            done=terminal,
        )

    def _enn_action_to_doom(self, action: Mapping[str, Action]) -> List[int]:
        doom_action = [0 for _ in range(len(self._available_buttons))]
        for i, button in enumerate(self._available_buttons):
            # There is only one actor (player)
            button_action = action[button.name].indices[0]  # type: ignore
            if "DELTA" in button.name:
                doom_action[i] = DELTA_SPEED_STEPS[button_action]
            else:
                doom_action[i] = button_action
        return doom_action

    def act(self, actions: Mapping[str, Action]) -> Observation:
        doom_action = self._enn_action_to_doom(actions)
        reward = self._doomgame.make_action(doom_action, self._frame_skip)
        terminal = self._doomgame.is_episode_finished()
        if terminal:
            # No observation available,
            # give the previous observation
            observation = self._last_observation
            observation.done = True
        else:
            state = self._doomgame.get_state()
            self._last_state = state
            observation = self._build_state(state, reward, terminal)
        # Keep track of the last_observation
        # in case we hit end of the episode
        # (no state available, give last_observation instead)
        self._last_observation = observation
        return observation

    def initialize(self) -> None:
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
        self._last_state = state
        observation = self._build_state(state, reward=0.0, terminal=False)
        self._last_observation = observation
        return observation

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        if "mode" in kwargs and kwargs["mode"] == "rgb_array":
            if self._last_state is not None:
                return self._last_state.screen_buffer.transpose([1, 2, 0])
            else:
                return np.zeros(
                    (self._image_height, self._image_width, 3), dtype=np.uint8
                )
        else:
            raise ValueError(
                "Only 'rgb_array' render mode is supported (`render(mode='rgb_mode')`)"
            )

    def __del__(self) -> None:
        self._doomgame.close()
