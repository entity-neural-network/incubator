from typing import Any, Dict
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
