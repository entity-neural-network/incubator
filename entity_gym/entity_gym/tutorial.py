from typing import Dict, Mapping

from entity_gym.cli_runner import CliRunner
from entity_gym.environment import *
from entity_gym.environment.environment import (
    GlobalCategoricalAction,
    GlobalCategoricalActionMask,
    GlobalCategoricalActionSpace,
)


class TreasureHunt(Environment):
    def __init__(self) -> None:
        self.x_pos = 0
        self.y_pos = 0

    def obs_space(self) -> ObsSpace:
        return ObsSpace(global_features=["x_pos", "y_pos"])

    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(
            done=False,
            reward=0,
            global_features=[self.x_pos, self.y_pos],
            actions={"move": GlobalCategoricalActionMask()},
        )

    def action_space(self) -> Dict[str, ActionSpace]:
        return {
            "move": GlobalCategoricalActionSpace(
                choices=["up", "down", "left", "right"]
            )
        }

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        action = actions["move"]
        assert isinstance(action, GlobalCategoricalAction)
        if action.label == "up":
            self.y_pos += 1
        elif action.label == "down":
            self.y_pos -= 1
        elif action.label == "left":
            self.x_pos -= 1
        elif action.label == "right":
            self.x_pos += 1
        return self._observe()


if __name__ == "__main__":
    env = TreasureHunt()
    # env = ENV_REGISTRY["MultiSnake"]()
    # env = ENV_REGISTRY["CherryPick"]()
    CliRunner(env).run()
