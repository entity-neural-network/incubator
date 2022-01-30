import random
import numpy as np
from typing import Dict, List, Mapping, Tuple
from entity_gym.environment import *


class MineSweeper(Environment):
    """
    The MineSweeper environment contains two types of objects, mines and robots.
    The player controls all robots in the environment.
    On every step, each robot may move in one of four cardinal directions, or stay in place and defuse all adjacent mines.
    If a robot defuses a mine, it is removed from the environment.
    If a robot steps on a mine, it is removed from the environment and the player loses the game.
    The player wins the game when all mines are defused.
    """

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        nmines: int = 5,
        nrobots: int = 2,
    ):
        self.width = width
        self.height = height
        self.nmines = nmines
        self.nrobots = nrobots
        # Positions of robots and mines
        self.robots: List[Tuple[int, int]] = []
        self.mines: List[Tuple[int, int]] = []

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Mine": Entity(features=["x", "y"]),
                "Robot": Entity(features=["x", "y"]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[ActionType, ActionSpace]:
        return {
            "Move": CategoricalActionSpace(
                ["Up", "Down", "Left", "Right", "Defuse Mines"],
            ),
        }

    def reset(self) -> Observation:
        positions = random.sample(
            [(x, y) for x in range(self.width) for y in range(self.height)],
            self.nmines + self.nrobots,
        )
        self.mines = positions[: self.nmines]
        self.robots = positions[self.nmines :]
        return self.observe()

    def observe(self) -> Observation:
        done = len(self.mines) == 0 or len(self.robots) == 0
        reward = 1.0 if len(self.mines) == 0 else 0.0
        return Observation(
            features={
                "Mine": np.array(
                    self.mines,
                    dtype=np.float32,
                ).reshape(-1, 2),
                "Robot": np.array(
                    self.robots,
                    dtype=np.float32,
                ).reshape(-1, 2),
            },
            ids={
                # Identifiers for each Robot
                "Robot": [("Robot", i) for i in range(len(self.robots))],
                # We don't need identifiers for mines since they are not
                # directly referenced by any actions.
            },
            actions={
                "Move": DenseCategoricalActionMask(
                    # Allow all robots to move
                    actor_types=["Robot"],
                ),
            },
            # The game is done once there are no more mines or robots
            done=done,
            # Give reward of 1.0 for defusing all mines
            reward=reward,
            end_of_episode_info=EpisodeStats(10, reward) if done else None,
        )

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        move = actions["Move"]
        assert isinstance(move, CategoricalAction)
        for (_, i), choice in move.items():
            # Action space is ["Up", "Down", "Left", "Right", "Defuse Mines"],
            x, y = self.robots[i]
            if choice == 0 and y < self.height - 1:
                self.robots[i] = (x, y + 1)
            elif choice == 1 and y > 0:
                self.robots[i] = (x, y - 1)
            elif choice == 2 and x > 0:
                self.robots[i] = (x - 1, y)
            elif choice == 3 and x < self.width - 1:
                self.robots[i] = (x + 1, y)
            elif choice == 4:
                # Remove all mines adjacent to this robot
                rx, ry = self.robots[i]
                self.mines = [
                    (x, y) for (x, y) in self.mines if abs(x - rx) + abs(y - ry) > 1
                ]

        # Remove all robots that stepped on a mine
        self.robots = [(x, y) for (x, y) in self.robots if (x, y) not in self.mines]

        return self.observe()
