import random
import numpy as np
from typing import Dict, List, Mapping, Optional, Tuple
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
        orbital_cannon: bool = False,
        cooldown_period: int = 5,
    ):
        self.width = width
        self.height = height
        self.nmines = nmines
        self.nrobots = nrobots
        self.orbital_cannon = orbital_cannon
        self.cooldown_period = cooldown_period
        self.orbital_cannon_cooldown = cooldown_period
        # Positions of robots and mines
        self.robots: List[Optional[Tuple[int, int]]] = []
        self.mines: List[Tuple[int, int]] = []

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Mine": Entity(features=["x", "y"]),
                "Robot": Entity(features=["x", "y"]),
                "Orbital Cannon": Entity(["cooldown"]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[ActionType, ActionSpace]:
        return {
            "Move": CategoricalActionSpace(
                ["Up", "Down", "Left", "Right", "Defuse Mines"],
            ),
            "Fire Orbital Cannon": SelectEntityActionSpace(),
        }

    def reset(self) -> Observation:
        positions = random.sample(
            [(x, y) for x in range(self.width) for y in range(self.height)],
            self.nmines + self.nrobots,
        )
        self.mines = positions[: self.nmines]
        self.robots = list(positions[self.nmines :])
        self.orbital_cannon_cooldown = self.cooldown_period
        return self.observe()

    def observe(self) -> Observation:
        done = len(self.mines) == 0 or len(self.robots) == 0
        reward = 1.0 if len(self.mines) == 0 else 0.0
        return Observation.from_entity_obs(
            entities={
                "Mine": EntityObs(
                    features=self.mines,
                    ids=[("Mine", i) for i in range(len(self.mines))],
                ),
                "Robot": EntityObs(
                    features=[r for r in self.robots if r is not None],
                    ids=[("Robot", i) for i in range(len(self.robots))],
                ),
                "Orbital Cannon": EntityObs(
                    features=[(self.orbital_cannon_cooldown,)],
                    ids=[("Orbital Cannon", 0)],
                )
                if self.orbital_cannon
                else None,
            },
            actions={
                "Move": CategoricalActionMask(
                    # Allow all robots to move
                    actor_types=["Robot"],
                    mask=[self.valid_moves(*r) for r in self.robots if r is not None],
                ),
                "Fire Orbital Cannon": SelectEntityActionMask(
                    # Only the Orbital Cannon can fire, but not if cooldown > 0
                    actor_types=["Orbital Cannon"]
                    if self.orbital_cannon_cooldown == 0
                    else [],
                    # Both mines and robots can be fired at
                    actee_types=["Mine", "Robot"],
                ),
            },
            # The game is done once there are no more mines or robots
            done=done,
            # Give reward of 1.0 for defusing all mines
            reward=reward,
            end_of_episode_info=EpisodeStats(10, reward) if done else None,
        )

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        fire = actions["Fire Orbital Cannon"]
        assert isinstance(fire, SelectEntityAction)
        for (entity_type, i) in fire.actees:
            if entity_type == "Mine":
                self.mines.remove(self.mines[i])
            elif entity_type == "Robot":
                # Don't remove yet to keep indices valid
                self.robots[i] = None

        move = actions["Move"]
        assert isinstance(move, CategoricalAction)
        for (_, i), choice in move.items():
            if self.robots[i] is None:
                continue
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
        self.robots = [r for r in self.robots if r is not None and r not in self.mines]

        return self.observe()

    def valid_moves(self, x: int, y: int) -> List[bool]:
        return [
            x < self.width - 1,
            x > 0,
            y < self.height - 1,
            y > 0,
            # Always allow staying in place and defusing mines
            True,
        ]
