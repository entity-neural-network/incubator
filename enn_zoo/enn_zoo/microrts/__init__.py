from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import random
import numpy as np
from copy import deepcopy
import os
import gym_microrts
from gym_microrts import microrts_ai
import xml.etree.ElementTree as ET
import json

from entity_gym.environment import (
    CategoricalAction,
    DenseCategoricalActionMask,
    Entity,
    Environment,
    EpisodeStats,
    ObsSpace,
    Type,
    CategoricalActionSpace,
    ActionSpace,
    Observation,
    Action,
    VecEnv,
)

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import JArray, JInt


class GymMicrorts(Environment):
    """
    Turn-based version of Snake with multiple snakes.
    Each snake has a different color.
    For each snake, Food of that color is placed randomly on the board.
    Snakes can only eat Food of their color.
    When a snake eats Food of the same color, it grows by one unit.
    When a snake grows and it's length was less than 11, the player receives a reward of 0.1 / num_snakes.
    The game ends when a snake collides with another snake, runs into a wall, eats Food of another color, or all snakes reach a length of 11.
    """

    def __init__(
        self,
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        partial_obs=False,
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
    ):
        self.map_paths = map_paths
        self.partial_obs = partial_obs
        self.reward_weight = reward_weight

        self.step = 0

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], "microrts")
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar",
                "lib/bots/Coac.jar",
                "lib/bots/Droplet.jar",
                "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar",
                "lib/bots/MixedBot.jar",
                "lib/bots/TiamatBot.jar",
                "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar",  # "MindSeal.jar"
            ]
            for jar in jars:
                print(os.path.join(self.microrts_path, jar))
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable

        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import (
            RewardFunctionInterface,
            WinLossRewardFunction,
            ResourceGatherRewardFunction,
            AttackRewardFunction,
            ProduceWorkerRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            CloserToEnemyBaseRewardFunction,
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )

        self.ai2s = [microrts_ai.coacAI for _ in range(1)]

        from ts.entity import JNIEntityClient as Client
        from ai.core import AI

        self.client = Client(
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths[0],
            self.ai2s[0](self.real_utt),
            self.real_utt,
            self.partial_obs,
        )
        # get the unit type table
        self.utt = json.loads(str(self.client.sendUTT()))

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Resource": Entity(["x", "y"]),
                "Base": Entity(["x", "y"]),
                "Barracks": Entity(["x", "y"]),
                "Worker": Entity(["x", "y"]),
                "Light": Entity(["x", "y"]),
                "Heavy": Entity(["x", "y"]),
                "Ranged": Entity(["x", "y"]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "unitaction": CategoricalActionSpace(
                choices=["move_up", "move_right", "move_down", "move_left"],
            ),
        }

    def _reset(self) -> Observation:
        self.step = 0
        return self._observe()

    def _act(self, action: Mapping[str, Action]) -> Observation:
        game_over = False
        self.step += 1

        print("=========", action)
        if "unitaction" in action:
            response = self.client.gameStep(action["unitaction"].actions, 0)
        else:
            response = self.client.gameStep([], 0)
        self.client.render(False)
        return Observation(
            entities=self.generate_entities(response),
            ids=np.array(response.observation[7]),
            action_masks={
                "unitaction": DenseCategoricalActionMask(
                    actors=np.array(response.observation[8]),
                ),
            },
            reward=response.reward @ self.reward_weight,
            done=response.done[0],
            end_of_episode_info=EpisodeStats(length=self.step, total_reward=1)
            if response.done[0]
            else None,
        )

    def _observe(self, done: bool = False, player: int = 0) -> Observation:
        response = self.client.reset(0)
        self.client.render(False)

        print("===xx", np.array(response.observation[8]))
        print("===xx", np.array(response.observation[9]))
        return Observation(
            entities=self.generate_entities(response),
            ids=np.array(response.observation[7]),
            action_masks={
                "unitaction": DenseCategoricalActionMask(
                    actors=np.array(response.observation[8]),
                ),
            },
            reward=response.reward @ self.reward_weight,
            done=response.done[0],
            end_of_episode_info=EpisodeStats(length=self.step, total_reward=1)
            if response.done[0]
            else None,
        )

    def generate_entities(self, response):
        entities = {}
        resource = np.array(response.observation[0]).astype(np.float32)
        base = np.array(response.observation[1]).astype(np.float32)
        barracks = np.array(response.observation[2]).astype(np.float32)
        worker = np.array(response.observation[3]).astype(np.float32)
        light = np.array(response.observation[4]).astype(np.float32)
        heavy = np.array(response.observation[5]).astype(np.float32)
        ranged = np.array(response.observation[6]).astype(np.float32)
        if len(resource) > 0:
            entities["Resource"] = resource
        if len(base) > 0:
            entities["Base"] = base
        if len(barracks) > 0:
            entities["Barracks"] = barracks
        if len(worker) > 0:
            entities["Worker"] = worker
        if len(light) > 0:
            entities["Light"] = light
        if len(heavy) > 0:
            entities["Heavy"] = heavy
        if len(ranged) > 0:
            entities["Ranged"] = ranged
        
        print(entities)
        return entities
