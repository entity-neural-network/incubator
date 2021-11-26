from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import random
import numpy as np
from copy import deepcopy

from entity_gym.environment import (
    CategoricalAction,
    DenseCategoricalActionMask,
    Entity,
    Environment,
    ObsSpace,
    Type,
    CategoricalActionSpace,
    ActionSpace,
    Observation,
    Action,
    VecEnv,
)


@dataclass
class Snake:
    color: int
    segments: List[Tuple[int, int]]


@dataclass
class Food:
    color: int
    position: Tuple[int, int]


class MultiSnake(Environment):
    """
    Turn-based version of Snake with multiple snakes.
    Each snake has a different color.
    For each snake, Food of that color is placed randomly on the board.
    Snakes can only eat Food of their color.
    When a snake eats Food of the same color, it grows by one unit.
    When a snake grows and it's length was less than 11, the player receives a reward of 0.1 / num_snakes.
    The game ends when a snake collides with another snake, runs into a wall, eats Food of another color, or all snakes reach a length of 11.
    """

    def __init__(self, board_size: int = 10, num_snakes: int = 2, num_players: int = 1):
        """
        :param num_players: number of players
        :param board_size: size of the board
        :param num_snakes: number of snakes per player
        """
        assert num_snakes < 10, f"num_snakes must be less than 10, got {num_snakes}"
        self.board_size = board_size
        self.num_snakes = num_snakes
        self.num_players = num_players
        self.snakes: List[Snake] = []
        self.food: List[Food] = []
        self.game_over = False
        self.last_scores = [0] * self.num_players
        self.scores = [0] * self.num_players

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "SnakeHead": Entity(["x", "y", "color"]),
                "SnakeBody": Entity(["x", "y", "color"]),
                "Food": Entity(["x", "y", "color"]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "move": CategoricalActionSpace(
                choices=["up", "down", "left", "right"],
            ),
        }

    def _spawn_snake(self, color: int) -> None:
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            if any(
                (x, y) == (sx, sy) for snake in self.snakes for sx, sy in snake.segments
            ):
                continue
            self.snakes.append(Snake(color, [(x, y)]))
            break

    def _spawn_food(self, color: int) -> None:
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            if any((x, y) == (f.position[0], f.position[1]) for f in self.food) or any(
                (x, y) == (sx, sy) for snake in self.snakes for sx, sy in snake.segments
            ):
                continue
            self.food.append(Food(color, (x, y)))
            break

    def _reset(self) -> Observation:
        self.snakes = []
        self.food = []
        for i in range(self.num_snakes):
            self._spawn_snake(i)
        for i in range(self.num_snakes):
            self._spawn_food(i)
        return self._observe()

    def _act(self, action: Mapping[str, Action]) -> Observation:
        game_over = False
        reward = 0.0
        move_action = action["move"]
        self.last_scores = deepcopy(self.scores)
        assert isinstance(move_action, CategoricalAction)
        for id, move in move_action.actions:
            snake = self.snakes[id]
            x, y = snake.segments[-1]
            if move == 0:
                y += 1
            elif move == 1:
                y -= 1
            elif move == 2:
                x -= 1
            elif move == 3:
                x += 1
            if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
                game_over = True
            if any((x, y) == (sx, sy) for s in self.snakes for sx, sy in s.segments):
                game_over = True
            ate_Food = False
            for i in range(len(self.food)):
                if self.food[i].position == (x, y):
                    if self.food[i].color != snake.color:
                        game_over = True
                    elif len(snake.segments) < 11:
                        ate_Food = True
                        self.scores[id // self.num_players] += 0.1 / self.num_snakes
                    self.food.pop(i)
                    self._spawn_food(snake.color)
                    break
            snake.segments.append((x, y))
            if not ate_Food:
                snake.segments = snake.segments[1:]
        for player in range(self.num_players):
            snakes_per_player = self.num_snakes // self.num_players
            if all(
                len(s.segments) >= 11
                for s in self.snakes[
                    player * snakes_per_player : (player + 1) * snakes_per_player
                ]
            ):
                game_over = True
        return self._observe(done=game_over)

    def _observe(self, done: bool = False, player: int = 0) -> Observation:
        color_offset = player * (self.num_snakes // self.num_players)

        def cycle_color(color: int) -> int:
            return (color - color_offset) % self.num_snakes

        return Observation(
            entities={
                "SnakeHead": np.array(
                    [
                        [
                            s.segments[0][0],
                            s.segments[0][1],
                            cycle_color(s.color),
                        ]
                        for s in self.snakes
                    ]
                ),
                "SnakeBody": np.array(
                    [
                        [sx, sy, cycle_color(snake.color)]
                        for snake in self.snakes
                        for sx, sy in snake.segments[1:]
                    ]
                ).reshape(-1, 3),
                "Food": np.array(
                    [
                        [
                            f.position[0],
                            f.position[1],
                            cycle_color(f.color),
                        ]
                        for f in self.food
                    ]
                ),
            },
            ids=list(
                range(sum([len(s.segments) for s in self.snakes]) + len(self.food))
            ),
            action_masks={
                "move": DenseCategoricalActionMask(
                    actors=np.arange(self.num_snakes),
                ),
            },
            reward=self.scores[player] - self.last_scores[player],
            done=done,
        )


class MultiplayerMultiSnake(VecEnv):
    """
    Multiplayer version of multi-snake.
    Implements VecEnv directly to allow for multiple players without requiring proper multi-agent support.
    Each player controls and receives rewards for a subset of the snakes.
    Only one player has to reach length 11 on all its snakes for the game to be over.
    """

    def __init__(
        self,
        board_size: int = 10,
        num_snakes: int = 4,
        num_players: int = 2,
        num_envs: int = 1,
    ):
        assert num_snakes % num_players == 0
        self.envs = [
            MultiSnake(board_size, num_snakes, num_players) for _ in range(num_envs)
        ]
        self.num_players = num_players
        self.num_snakes = num_snakes

    @classmethod
    def env_cls(cls) -> Type[Environment]:
        return MultiSnake

    def _reset(self) -> List[Observation]:
        obs = []
        for env in self.envs:
            o = env._reset()
            obs.append(o)
            obs.append(o)
        return obs

    def _act(self, actions: Sequence[Mapping[str, Action]]) -> List[Observation]:
        obs = []
        for i, env in enumerate(self.envs):
            acts = actions[i * self.num_players : (i + 1) * self.num_players]
            combined_acts = {}
            for act in acts:
                for k, v in act.items():
                    assert isinstance(v, CategoricalAction)
                    if k not in combined_acts:
                        combined_acts[k] = v
                    else:
                        combined_acts[k].actions.extend(v.actions)
            env._act(combined_acts)
            for j in range(self.num_players):
                obs.append(env._observe(player=j))
            if obs[0].done:
                env._reset()
        return obs
