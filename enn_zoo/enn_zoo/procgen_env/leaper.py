from typing import Dict, List

import numpy as np

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_int(bottom_road_y);
# b->write_vector_float(road_lane_speeds);
# b->write_int(bottom_water_y);
# b->write_vector_float(water_lane_speeds);
# b->write_int(goal_y);

LEAPER_FEATS = [
    "bottom_road_y",
    "road_lane_speed0",
    "road_lane_speed1",
    "road_lane_speed2",
    "road_lane_speed3",
    "road_lane_speed4",
    "bottom_water_y",
    "water_lane_speed0",
    "water_lane_speed1",
    "water_lane_speed2",
    "water_lane_speed3",
    "water_lane_speed4",
    "goal_y",
]


class Leaper(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("leaper", distribution_mode)

    def _global_feats(self) -> List[str]:
        return LEAPER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        bottom_road_y = float(data.read_int())
        road_lane_speeds = data.read_float_array()
        bottom_water_y = float(data.read_int())
        water_lane_speeds = data.read_float_array()
        goal_y = float(data.read_int())
        assert len(road_lane_speeds) <= 5
        assert len(water_lane_speeds) <= 5
        road_lane_speeds = np.pad(
            road_lane_speeds,
            (0, 5 - len(road_lane_speeds)),
            "constant",
            constant_values=0.0,
        )
        water_lane_speeds = np.pad(
            water_lane_speeds,
            (0, 5 - len(water_lane_speeds)),
            "constant",
            constant_values=0.0,
        )
        return [
            bottom_road_y,
            *road_lane_speeds,
            bottom_water_y,
            *water_lane_speeds,
            goal_y,
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int LOG = 1;
        # const int ROAD = 2;
        # const int WATER = 3;
        # const int CAR = 4;
        # const int FINISH_LINE = 5;
        return {
            1: "Log",
            2: "Road",
            3: "Water",
            4: "Car",
            5: "FinishLine",
        }
