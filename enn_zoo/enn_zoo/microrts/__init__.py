from dataclasses import dataclass
from tokenize import String
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import random
from entity_gym.environment.environment import EntityObs
import numpy as np
import numpy.typing as npt
from copy import deepcopy
import os
import gym_microrts
from gym_microrts import microrts_ai
import xml.etree.ElementTree as ET
import json

from entity_gym.environment import (
    CategoricalAction,
    CategoricalActionMask,
    Entity,
    Environment,
    EpisodeStats,
    ObsSpace,
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
        map_paths: List[str] = ["maps/10x10/basesTwoWorkers10x10.xml"],
        partial_obs: bool = False,
        reward_weight: npt.NDArray[np.float32] = np.array(
            [0.0, 1.0, 0.0, 0.0, 0.0, 5.0]
        ),
    ):
        self.map_paths = map_paths
        self.partial_obs = partial_obs
        self.reward_weight = reward_weight

        self.step = 0

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], "microrts")
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))  # type: ignore

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
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
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
            "unit_action": CategoricalActionSpace(
                choices=[
                    "move_up",
                    "move_right",
                    "move_down",
                    "move_left",
                    "harvest_up",
                    "harvest_right",
                    "harvest_down",
                    "harvest_left",
                    "return_up",
                    "return_right",
                    "return_down",
                    "return_left",
                    "produce_base_up",
                    "produce_base_right",
                    "produce_base_down",
                    "produce_base_left",
                    "produce_barrack_up",
                    "produce_barrack_right",
                    "produce_barrack_down",
                    "produce_barrack_left",
                ]
                + [
                    f"attack_location_{i}" for i in range(49)
                ],  # the attack trange is a 7x7 relative grid
            ),
            "base_action": CategoricalActionSpace(
                choices=[
                    "produce_worker_up",
                    "produce_worker_right",
                    "produce_worker_down",
                    "produce_worker_left",
                ],
            ),
            "barrack_action": CategoricalActionSpace(
                choices=[
                    "produce_light_up",
                    "produce_light_right",
                    "produce_light_down",
                    "produce_light_left",
                    "produce_heavy_up",
                    "produce_heavy_right",
                    "produce_heavy_down",
                    "produce_heavy_left",
                    "produce_ranged_up",
                    "produce_ranged_right",
                    "produce_ranged_down",
                    "produce_ranged_left",
                ],
            ),
        }

    def reset(self) -> Observation:
        self.step = 0
        self.total_reward = 0

        response = self.client.reset(0)
        self.client.render(False)

        unit_action_actor_ids = np.array(response.observation[8])
        unit_action_actor_masks = np.array(response.observation[9], dtype=np.bool8)
        base_action_actor_ids = np.array(response.observation[10])
        base_action_actor_masks = np.array(response.observation[11], dtype=np.bool8)
        barrack_action_actor_ids = np.array(response.observation[12])
        barrack_action_actor_masks = np.array(response.observation[13], dtype=np.bool8)
        return Observation.from_entity_obs(
            entities=self.generate_entities(response),
            actions={
                "unit_action": CategoricalActionMask(
                    actor_ids=unit_action_actor_ids,  # type: ignore
                    mask=unit_action_actor_masks,
                ),
                "base_action": CategoricalActionMask(
                    actor_ids=base_action_actor_ids,  # type: ignore
                    mask=base_action_actor_masks,
                ),
                "barrack_action": CategoricalActionMask(
                    actor_ids=barrack_action_actor_ids,  # type: ignore
                    mask=barrack_action_actor_masks,
                ),
            },
            reward=response.reward @ self.reward_weight,
            done=response.done[0],
            end_of_episode_info=EpisodeStats(length=self.step, total_reward=1)
            if response.done[0]
            else None,
        )

    def act(self, action: Mapping[str, Action]) -> Observation:
        game_over = False
        self.step += 1

        unit_action_actors: Sequence[Any] = []
        unit_actions: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        base_action_actors: Sequence[Any] = []
        base_actions: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        barrack_action_actors: Sequence[Any] = []
        barrack_actions: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        if "unit_action" in action and isinstance(
            action["unit_action"], CategoricalAction
        ):
            unit_action_actors = action["unit_action"].actors
            unit_actions = action["unit_action"].actions
        if "base_action" in action and isinstance(
            action["base_action"], CategoricalAction
        ):
            base_action_actors = action["base_action"].actors
            base_actions = action["base_action"].actions
        if "barrack_action" in action and isinstance(
            action["barrack_action"], CategoricalAction
        ):
            barrack_action_actors = action["barrack_action"].actors
            barrack_actions = action["barrack_action"].actions

        response = self.client.gameStep(
            unit_action_actors,
            unit_actions,
            base_action_actors,
            base_actions,
            barrack_action_actors,
            barrack_actions,
            0,
        )

        self.client.render(False)
        unit_action_actor_ids = np.array(response.observation[8])
        unit_action_actor_masks = None
        if len(unit_action_actor_ids) > 0:
            unit_action_actor_masks = np.array(response.observation[9], dtype=np.bool8)
        base_action_actor_ids = np.array(response.observation[10])
        base_action_actor_masks = None
        if len(base_action_actor_ids) > 0:
            base_action_actor_masks = np.array(response.observation[11], dtype=np.bool8)
        barrack_action_actor_ids = np.array(response.observation[12])
        barrack_action_actor_masks = None
        if len(barrack_action_actor_ids) > 0:
            barrack_action_actor_masks = np.array(
                response.observation[13], dtype=np.bool8
            )

        self.total_reward += response.reward @ self.reward_weight
        return Observation.from_entity_obs(
            entities=self.generate_entities(response),
            actions={
                "unit_action": CategoricalActionMask(
                    actor_ids=unit_action_actor_ids,  # type: ignore
                    mask=unit_action_actor_masks,
                ),
                "base_action": CategoricalActionMask(
                    actor_ids=base_action_actor_ids,  # type: ignore
                    mask=base_action_actor_masks,
                ),
                "barrack_action": CategoricalActionMask(
                    actor_ids=barrack_action_actor_ids,  # type: ignore
                    mask=barrack_action_actor_masks,
                ),
            },
            reward=response.reward @ self.reward_weight,
            done=response.done[0],
            end_of_episode_info=EpisodeStats(
                length=self.step, total_reward=self.total_reward
            )
            if response.done[0]
            else None,
        )

    def generate_entities(self, response: Any) -> Mapping[str, Optional[EntityObs]]:
        entities: MutableMapping[str, Optional[EntityObs]] = {}
        resource = np.array(response.observation[0]).astype(np.float32)
        base = np.array(response.observation[1]).astype(np.float32)
        barracks = np.array(response.observation[2]).astype(np.float32)
        worker = np.array(response.observation[3]).astype(np.float32)
        light = np.array(response.observation[4]).astype(np.float32)
        heavy = np.array(response.observation[5]).astype(np.float32)
        ranged = np.array(response.observation[6]).astype(np.float32)
        entity_ids = list(np.array(response.observation[7]))  # type: Sequence[Any]
        if len(resource) > 0:
            entities["Resource"] = EntityObs(
                features=resource[:, 1:], ids=resource[:, 0].astype(np.int32)  # type: ignore
            )
        if len(base) > 0:
            entities["Base"] = EntityObs(
                features=base[:, 1:], ids=base[:, 0].astype(np.int32)  # type: ignore
            )
        if len(barracks) > 0:
            entities["Barracks"] = EntityObs(
                features=barracks[:, 1:], ids=barracks[:, 0].astype(np.int32)  # type: ignore
            )
        if len(worker) > 0:
            entities["Worker"] = EntityObs(
                features=worker[:, 1:], ids=worker[:, 0].astype(np.int32)  # type: ignore
            )
        if len(light) > 0:
            entities["Light"] = EntityObs(
                features=light[:, 1:], ids=light[:, 0].astype(np.int32)  # type: ignore
            )
        if len(heavy) > 0:
            entities["Heavy"] = EntityObs(
                features=heavy[:, 1:], ids=heavy[:, 0].astype(np.int32)  # type: ignore
            )
        if len(ranged) > 0:
            entities["Ranged"] = EntityObs(
                features=ranged[:, 1:], ids=ranged[:, 0].astype(np.int32)  # type: ignore
            )

        return entities

    def __del__(self) -> None:
        self.client.close()
