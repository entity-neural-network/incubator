import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import gym_microrts
import jpype
import jpype.imports
import numpy as np
import numpy.typing as npt
from gym_microrts import microrts_ai
from jpype.imports import registerDomain
from jpype.types import JArray
from PIL import Image

from entity_gym.environment import (
    Action,
    ActionSpace,
    CategoricalAction,
    CategoricalActionMask,
    CategoricalActionSpace,
    Entity,
    Environment,
    Observation,
    ObsSpace,
)
from entity_gym.environment.environment import EntityObs


class GymMicrorts(Environment):
    """
    A real-time strategy environment for microrts.
    See https://github.com/santiontanon/microrts

    Light grey squares are bases, dark grey squares are barracks,
    green squares are resources, colored circles are combat units,
    and grey circles are workers that harvest resources.

    Args:
        map_path: the path to the map, see the list of supported maps [here](https://github.com/vwxyzjn/microrts/tree/52d17e58592722889197aeee03fffafb154cfb8c/maps)
        reward_weight: the weight mutiplied to each each reward functions,
            which are in order:
            - win/loss reward: + 1 for win, - 1 for loss, 0 for tie
            - resource gather reward: + 1 for each resource gathered and +1 for returned
            - produce worker reward: + 1 for each worker produced
            - produce building reward: + 1 for each building produced
            - attack reward: + 1 for each attack action
            - produce combat unit reward: + 1 for each combat unit produced
    """

    def __init__(
        self,
        map_path: str = "maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight: List[float] = [10.0, 1.0, 1.0, 0.2, 1.0, 4.0],
    ):
        self.map_path = map_path
        self.reward_weight = np.array(reward_weight)
        self.step = 0

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], "microrts")
        root = ET.parse(os.path.join(self.microrts_path, self.map_path)).getroot()
        self.height = int(root.get("height"))  # type: ignore
        self.width = int(root.get("width"))  # type: ignore

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
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            WinLossRewardFunction,
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
        self.rfs_names = [str(rf) for rf in self.rfs]

        self.ai2s = [microrts_ai.coacAI for _ in range(1)]

        from ts.entity import JNIEntityClient as Client

        self.client = Client(
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_path,
            self.ai2s[0](self.real_utt),
            self.real_utt,
            False,
        )
        # get the unit type table
        self.utt = json.loads(str(self.client.sendUTT()))

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            entities={
                "Resource": Entity(["x", "y"]),
                "Base": Entity(["x", "y"]),
                "Barracks": Entity(["x", "y"]),
                "Worker": Entity(["x", "y"]),
                "Light": Entity(["x", "y"]),
                "Heavy": Entity(["x", "y"]),
                "Ranged": Entity(["x", "y"]),
            }
        )

    def action_space(self) -> Dict[str, ActionSpace]:
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

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        if "mode" in kwargs and kwargs["mode"] == "rgb_array":
            bytes_array = np.array(self.client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)[:, :, ::-1]
        else:
            return self.client.render(False)  # type: ignore

    def reset(self) -> Observation:
        self.step = 0
        self.returns = np.zeros(len(self.rfs))

        response = self.client.reset(0)

        unit_action_actor_ids = np.array(response.observation[8])
        unit_action_actor_masks = np.array(response.observation[9], dtype=np.bool8)
        base_action_actor_ids = np.array(response.observation[10])
        base_action_actor_masks = np.array(response.observation[11], dtype=np.bool8)
        barrack_action_actor_ids = np.array(response.observation[12])
        barrack_action_actor_masks = np.array(response.observation[13], dtype=np.bool8)
        return Observation(
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
        )

    def act(self, actions: Mapping[str, Action]) -> Observation:
        self.step += 1

        unit_action_actors: Sequence[Any] = []
        unit_actions: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        base_action_actors: Sequence[Any] = []
        base_actions: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        barrack_action_actors: Sequence[Any] = []
        barrack_actions: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        if "unit_action" in actions and isinstance(
            actions["unit_action"], CategoricalAction
        ):
            unit_action_actors = actions["unit_action"].actors
            unit_actions = actions["unit_action"].actions
        if "base_action" in actions and isinstance(
            actions["base_action"], CategoricalAction
        ):
            base_action_actors = actions["base_action"].actors
            base_actions = actions["base_action"].actions
        if "barrack_action" in actions and isinstance(
            actions["barrack_action"], CategoricalAction
        ):
            barrack_action_actors = actions["barrack_action"].actors
            barrack_actions = actions["barrack_action"].actions

        response = self.client.gameStep(
            unit_action_actors,
            unit_actions,
            base_action_actors,
            base_actions,
            barrack_action_actors,
            barrack_actions,
            0,
        )

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
        self.returns += response.reward

        return Observation(
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
            metrics=dict(
                zip(
                    [f"charts/episodic_return/{item}" for item in self.rfs_names],
                    self.returns,
                )
            ),
        )

    def generate_entities(self, response: Any) -> Mapping[str, Optional[EntityObs]]:
        entities: MutableMapping[str, Optional[EntityObs]] = {}
        for entity_type, observation in zip(
            ["Resource", "Base", "Barracks", "Worker", "Light", "Heavy", "Ranged"],
            response.observation,
        ):
            observation = np.array(observation).astype(np.float32)
            if len(observation) > 0:
                entities[entity_type] = EntityObs(
                    features=observation[:, 1:], ids=observation[:, 0].astype(np.int32)
                )
        return entities

    def __del__(self) -> None:
        self.client.close()
