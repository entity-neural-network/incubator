from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_int(last_fire_time);
# b->write_vector_bool(lane_directions);
# b->write_vector_bool(target_bools);
# b->write_vector_int(image_permutation);
# b->write_vector_float(lane_vels);
# b->write_int(num_lanes);
# b->write_int(num_current_ship_types);
# b->write_int(targets_hit);
# b->write_int(target_quota);
# b->write_float(juice_left);
# b->write_float(r_scale);
# b->write_float(spawn_prob);
# b->write_float(legend_r);
# b->write_float(min_agent_x);

PLUNDER_FEATS = [
    "last_fire_time",
    "lane_direction0",
    "lane_direction1",
    "lane_direction2",
    "lane_direction3",
    "lane_direction4",
    "target_bool0",
    "target_bool1",
    "target_bool2",
    "target_bool3",
    "target_bool4",
    "target_bool5",
    "lane_vel0",
    "lane_vel1",
    "lane_vel2",
    "lane_vel3",
    "lane_vel4",
    "num_lanes",
    "num_current_ship_types",
    "targets_hit",
    "target_quota",
    "juice_left",
    "r_scale",
    "spawn_prob",
    "legend_r",
    "min_agent_x",
]


class Plunder(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("plunder", distribution_mode)

    def _global_feats(self) -> List[str]:
        return PLUNDER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        last_fire_time = float(data.read_int())
        lane_directions = data.read_int_array()
        target_bools = data.read_int_array()
        data.read_int_array()
        lane_vels = data.read_float_array()
        num_lanes = float(data.read_int())
        num_current_ship_types = float(data.read_int())
        targets_hit = float(data.read_int())
        target_quota = float(data.read_int())
        juice_left = data.read_float()
        r_scale = data.read_float()
        spawn_prob = data.read_float()
        legend_r = data.read_float()
        min_agent_x = data.read_float()
        assert len(lane_directions) == 5
        assert len(lane_vels) == 5
        assert len(target_bools) == 6
        return [
            last_fire_time,
            *lane_directions,
            *target_bools,
            *lane_vels,
            num_lanes,
            num_current_ship_types,
            targets_hit,
            target_quota,
            juice_left,
            r_scale,
            spawn_prob,
            legend_r,
            min_agent_x,
        ]

    def _entity_types(self) -> Dict[int, str]:
        return {
            1: "PlayerBullet",
            2: "TargetLegend",
            3: "TargetBackground",
            6: "Panel",
            7: "Ship",
            54: "???",
        }
