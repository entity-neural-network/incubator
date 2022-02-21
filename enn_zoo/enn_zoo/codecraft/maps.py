import math
from typing import Any, Dict, List, Tuple, Union
import numpy as np


def map_allied_wealth(
    randomize: bool, hardness: int, require_default_mothership: bool
) -> Dict[str, Any]:
    map_width = 6000
    map_height = 3750
    mineral_count = 25
    angle = 2 * np.pi * np.random.rand()
    spawn_x = (map_width // 2 - 100) * np.sin(angle)
    spawn_y = (map_height // 2 - 100) * np.cos(angle)

    return {
        "mapWidth": map_width,
        "mapHeight": map_height,
        "minerals": mineral_count * [(1, 1)],
        "player1Drones": [
            drone_dict(
                spawn_x,
                spawn_y,
                constructors=2,
                storage_modules=4,
                engines=4,
                resources=0,
            )
        ],
        "player2Drones": [drone_dict(-spawn_x, -spawn_y, shield_generators=10)],
    }


def map_arena_tiny(
    randomize: bool, hardness: int, require_default_mothership: bool
) -> Dict[str, Any]:
    storage_modules = 1
    constructors = 1
    missiles_batteries = 1
    if randomize:
        storage_modules = np.random.randint(1, 3)
        constructors = np.random.randint(1, 3)
        missiles_batteries = np.random.randint(1, 3)
    return {
        "mapWidth": 1500,
        "mapHeight": 1500,
        "minerals": [],
        "player1Drones": [
            drone_dict(
                np.random.randint(-450, 450),
                np.random.randint(-450, 450),
                storage_modules=storage_modules,
                constructors=constructors,
            )
        ],
        "player2Drones": [
            drone_dict(
                np.random.randint(-450, 450),
                np.random.randint(-450, 450),
                missile_batteries=missiles_batteries,
                shield_generators=4 - missiles_batteries,
            )
        ],
    }


def drone_dict(
    x: int,
    y: int,
    storage_modules: int = 0,
    missile_batteries: int = 0,
    constructors: int = 0,
    engines: int = 0,
    shield_generators: int = 0,
    long_range_missiles: int = 0,
    resources: int = 0,
) -> Dict[str, int]:
    return {
        "xPos": x,
        "yPos": y,
        "resources": resources,
        "storageModules": storage_modules,
        "missileBatteries": missile_batteries,
        "constructors": constructors,
        "engines": engines,
        "shieldGenerators": shield_generators,
        "longRangeMissiles": long_range_missiles,
    }


def map_enhanced(
    randomize: bool, hardness: Union[int, float], require_default_mothership: bool
) -> Dict[str, Any]:
    if randomize:
        area = math.sqrt(np.random.uniform(1, (3 + hardness) ** 2))
    else:
        area = hardness

    eligible = [
        (x, y)
        for y in range(1, 20)
        for x in range(y, y * 2 + 1)
        if area // 2 <= x * y <= area
    ]
    x, y = eligible[np.random.randint(0, len(eligible))]
    map_width = 750 * x
    map_height = 750 * y
    mineral_count = 2 + np.random.randint(0, math.ceil(math.sqrt(x * y) / 5) + 2)
    minerals = [(1, np.random.randint(100, 1000)) for _ in range(mineral_count)]

    player1, player2 = enhanced_starting_drones(
        map_height, map_width, randomize and not require_default_mothership
    )
    return {
        "mapWidth": map_width,
        "mapHeight": map_height,
        "minerals": minerals,
        "player1Drones": player1,
        "player2Drones": player2,
    }


def enhanced_starting_drones(
    map_height: int, map_width: int, randomize: bool
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    drones = []
    starting_resources = np.random.randint(4, 8) if randomize else 7
    drones.append(
        dict(
            constructors=2,
            storage_modules=2,
            missile_batteries=1,
            shield_generators=1,
            engines=2,
            long_range_missiles=2,
            resources=2 * starting_resources,
        )
    )
    if randomize and np.random.uniform(0, 1) < 0.6:
        mstype = np.random.randint(0, 6)
        if mstype == 0:
            drones.append(
                dict(
                    constructors=1,
                    storage_modules=2,
                    engines=1,
                    resources=2 * starting_resources,
                )
            )
        elif mstype == 1:
            drones.append(
                dict(constructors=1, storage_modules=1, resources=starting_resources)
            )
        elif mstype == 2 or mstype == 3:
            drones.append(dict(storage_modules=1, resources=starting_resources))
        elif mstype == 4 or mstype == 5:
            drones.append(dict(storage_modules=2, resources=2 * starting_resources))

    angle = 2 * np.pi * np.random.rand()
    spawn_x = (map_width // 2 - 100) * np.sin(angle)
    spawn_y = (map_height // 2 - 100) * np.cos(angle)
    dcount = len(drones)
    if dcount == 1:
        spawn_offsets = [(0.0, 0.0)]
    else:
        spawn_offsets = [
            (
                (np.random.uniform(5, 30) ** 2) * np.sin(2 * math.pi * i / dcount),
                (np.random.uniform(5, 30) ** 2) * np.cos(2 * math.pi * i / dcount),
            )
            for i in range(dcount)
        ]

    def clip_x(x: int) -> int:
        return min(max(x, -map_width // 2), map_width // 2)

    def clip_y(y: int) -> int:
        return min(max(y, -map_height // 2), map_height // 2)

    player1 = [
        drone_dict(clip_x(spawn_x + x), clip_y(spawn_y + y), **ms)
        for ms, (x, y) in zip(drones, spawn_offsets)
    ]
    player2 = [
        drone_dict(clip_x(-spawn_x - x), clip_y(-spawn_y - y), **ms)
        for ms, (x, y) in zip(drones, spawn_offsets)
    ]
    return player1, player2
