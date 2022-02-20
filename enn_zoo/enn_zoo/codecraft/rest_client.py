import requests
import logging
import time
import os

import orjson
import numpy as np

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict


RETRIES = 100


@dataclass
class ObsConfig:
    allies: int
    drones: int
    minerals: int
    tiles: int
    num_builds: int
    relative_positions: bool = False
    feat_last_seen: bool = False
    feat_map_size: bool = True
    feat_is_visible: bool = True
    feat_abstime: bool = True
    v2: bool = True
    feat_rule_msdm: bool = True
    feat_rule_costs: bool = True
    feat_mineral_claims: bool = False
    harvest_action: bool = False
    lock_build_action: bool = False
    feat_dist_to_wall: bool = True
    unit_count: bool = True
    construction_progress: bool = True

    def global_features(self) -> int:
        gf = 2
        if self.feat_map_size:
            gf += 2
        if self.feat_abstime:
            gf += 2
        if self.feat_rule_msdm:
            gf += 1
        if self.feat_rule_costs:
            gf += self.num_builds
        if self.unit_count:
            gf += 1
        return gf

    def dstride(self) -> int:
        ds = 17
        if self.feat_last_seen:
            ds += 2
        if self.feat_is_visible:
            ds += 1
        if self.lock_build_action:
            ds += 1
        if self.feat_dist_to_wall:
            ds += 5
        if self.construction_progress:
            ds += self.num_builds + 2
        return ds

    def mstride(self) -> int:
        return 4 if self.feat_mineral_claims else 3

    def tstride(self) -> int:
        return 4

    def nonobs_features(self) -> int:
        return 5

    def enemies(self) -> int:
        return self.drones - self.allies

    def total_drones(self) -> int:
        return 2 * self.drones - self.allies

    def stride(self) -> int:
        return (
            self.global_features()
            + self.total_drones() * self.dstride()
            + self.minerals * self.mstride()
            + self.tiles * self.tstride()
        )

    def endglobals(self) -> int:
        return self.global_features()

    def endallies(self) -> int:
        return self.global_features() + self.dstride() * self.allies

    def endenemies(self) -> int:
        return self.global_features() + self.dstride() * self.drones

    def endmins(self) -> int:
        return self.endenemies() + self.mstride() * self.minerals

    def endtiles(self) -> int:
        return self.endmins() + self.tstride() * self.tiles

    def endallenemies(self) -> int:
        return self.endtiles() + self.dstride() * self.enemies()

    def extra_actions(self) -> int:
        if self.lock_build_action:
            return 2
        else:
            return 0


@dataclass
class Rules:
    mothership_damage_multiplier: float
    cost_modifiers: Dict[Tuple[int, int, int, int, int, int], float]


def create_game(
    game_length: Optional[int] = None,
    action_delay: int = 0,
    self_play: bool = False,
    custom_map: Optional[Dict[str, Any]] = None,
    scripted_opponent: str = "none",
    rules: Optional[Rules] = None,
    allowHarvesting: bool = True,
    forceHarvesting: bool = True,
    randomizeIdle: bool = True,
) -> int:
    assert rules is not None
    json = {
        "map": [] if custom_map is None else [custom_map],
        "costModifiers": list(rules.cost_modifiers.items()),
    }
    try:
        if game_length:
            response = requests.post(
                f"http://{get_hostname()}:9000/start-game"
                f"?maxTicks={game_length}"
                f"&actionDelay={action_delay}"
                f"&scriptedOpponent={scripted_opponent}"
                f"&mothershipDamageMultiplier={rules.mothership_damage_multiplier}"
                f"&allowHarvesting={scalabool(allowHarvesting)}"
                f"&forceHarvesting={scalabool(forceHarvesting)}"
                f"&randomizeIdle={scalabool(randomizeIdle)}",
                json=json,
            ).json()
        else:
            response = requests.post(
                f"http://{get_hostname()}:9000/start-game?actionDelay={action_delay}"
            ).json()
        return int(response["id"])
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on create_game, retrying")
        time.sleep(1)
        return create_game(
            game_length,
            action_delay,
            self_play,
            custom_map=custom_map,
            scripted_opponent=scripted_opponent,
            rules=rules,
            allowHarvesting=allowHarvesting,
            forceHarvesting=forceHarvesting,
            randomizeIdle=randomizeIdle,
        )


def act_batch(actions: List[Tuple[int, int, Any]]) -> None:
    payload = {}
    for game_id, player_id, player_actions in actions:
        player_actions_json = []
        for (
            move,
            turn,
            buildSpec,
            harvest,
            lockBuildAction,
            unlockBuildAction,
        ) in player_actions:
            player_actions_json.append(
                {
                    "buildDrone": buildSpec,
                    "move": move,
                    "harvest": harvest,
                    "transfer": False,
                    "turn": turn,
                    "lockBuildAction": lockBuildAction,
                    "unlockBuildAction": unlockBuildAction,
                }
            )
        payload[f"{game_id}.{player_id}"] = player_actions_json

    retries = 100
    while retries > 0:
        try:
            requests.post(
                f"http://{get_hostname()}:9000/batch-act",
                data=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
            ).raise_for_status()
            return
        except requests.exceptions.ConnectionError:
            # For some reason, a small percentage of requests fails with
            # "connection error (errno 98, address already in use)"
            # Just retry
            retries -= 1
            logging.info(f"Connection error on act_batch(), retrying")
            time.sleep(1)


def scalabool(b: bool) -> str:
    return "true" if b else "false"


def observe_batch_raw(
    obs_config: ObsConfig,
    game_ids: List[Tuple[int, int]],
    allies: int,
    drones: int,
    minerals: int,
    tiles: int,
    relative_positions: bool,
    v2: bool,
    extra_build_actions: List[Tuple[int, int, int, int, int, int]],
    map_size: bool = False,
    last_seen: bool = False,
    is_visible: bool = False,
    abstime: bool = False,
    rule_msdm: bool = False,
    rule_costs: bool = False,
    enforce_unit_cap: bool = False,
    unit_cap_override: int = 0,
) -> np.ndarray:
    retries = RETRIES
    url = (
        f"http://{get_hostname()}:9000/batch-observation?"
        f"json=false&"
        f"allies={allies}&"
        f"drones={drones}&"
        f"minerals={minerals}&"
        f"tiles={tiles}&"
        f"globalDrones=0&"
        f'relativePositions={"true" if relative_positions else "false"}&'
        f'lastSeen={"true" if last_seen else "false"}&'
        f'isVisible={"true" if is_visible else "false"}&'
        f'abstime={"true" if abstime else "false"}&'
        f'mapSize={"true" if map_size else "false"}&'
        f'v2={"true" if v2 else "false"}&'
        f"mineralClaims={scalabool(obs_config.feat_mineral_claims)}&"
        f"harvestAction={scalabool(obs_config.harvest_action)}&"
        f"ruleMsdm={scalabool(rule_msdm)}&"
        f"ruleCosts={scalabool(rule_costs)}&"
        f"lockBuildAction={scalabool(obs_config.lock_build_action)}&"
        f"distanceToWall={scalabool(obs_config.feat_dist_to_wall)}&"
        f"unitCount={scalabool(obs_config.unit_count)}&"
        f"enforceUnitCap={scalabool(enforce_unit_cap)}&"
        f"unitCapOverride={unit_cap_override}&"
        f"constructionProgress={scalabool(obs_config.construction_progress)}"
    )
    while retries > 0:
        json = [game_ids, extra_build_actions]
        try:
            response = requests.get(url, json=json, stream=True)
            response.raise_for_status()
            response_bytes = response.content
            return np.frombuffer(response_bytes, dtype=np.float32)  # type: ignore
        except requests.exceptions.ConnectionError as e:
            retries -= 1
            logging.info(f"Connection error on {url} with json={json}, retrying: {e}")
            time.sleep(10)
    raise RuntimeError(f"Failed to connect to {url} after {RETRIES} retries")


def get_hostname() -> str:
    xprun_id = os.getenv("XPRUN_ID")
    if xprun_id:
        return f"xprun.{xprun_id}.codecraftserver-0"
    else:
        return "localhost"


def observe_batch(game_ids: List[Tuple[int, int]]) -> Any:
    retries = RETRIES
    while retries > 0:
        try:
            return requests.get(
                f"http://{get_hostname()}:9000/batch-observation", json=[game_ids, []]
            ).json()
        except requests.exceptions.ConnectionError:
            retries -= 1
            logging.info(f"Connection error on observe_batch(), retrying")
            time.sleep(10)


def observe(game_id: int, player_id: int = 0) -> Any:
    try:
        return requests.get(
            f"http://{get_hostname()}:9000/observation?gameID={game_id}&playerID={player_id}"
        ).json()
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on observe({game_id}.{player_id}), retrying")
        time.sleep(1)
        return observe(game_id, player_id)


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)  # type: ignore


def dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy
