from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_vector_int(attack_modes);
# b->write_int(last_fire_time);
# b->write_int(time_to_swap);
# b->write_int(invulnerable_duration);
# b->write_int(vulnerable_duration);
# b->write_int(num_rounds);
# b->write_int(round_num);
# b->write_int(round_health);
# b->write_int(boss_vel_timeout);
# b->write_int(curr_vel_timeout);
# b->write_int(attack_mode);
# b->write_int(player_laser_theme);
# b->write_int(boss_laser_theme);
# b->write_int(damaged_until_time);
# b->write_bool(shields_are_up);
# b->write_bool(barriers_moves_right);
# b->write_float(base_fire_prob);
# b->write_float(boss_bullet_vel);
# b->write_float(barrier_vel);
# b->write_float(barrier_spawn_prob);
# b->write_float(rand_pct);
# b->write_float(rand_fire_pct);
# b->write_float(rand_pct_x);
# b->write_float(rand_pct_y);

# TODO: should all of these be exposed to policy?
BOSS_FIGHT_FEATS = [
    "last_fire_time",
    "time_to_swap",
    "invulnerable_duration",
    "vulnerable_duration",
    "num_rounds",
    "round_num",
    "round_health",
    "boss_vel_timeout",
    "curr_vel_timeout",
    "attack_mode",
    "player_laser_theme",
    "boss_laser_theme",
    "damaged_until_time",
    "shields_are_up",
    "barriers_moves_right",
    "base_fire_prob",
    "boss_bullet_vel",
    "barrier_vel",
    "barrier_spawn_prob",
    "rand_pct",
    "rand_fire_pct",
    "rand_pct_x",
    "rand_pct_y",
]


class BossFight(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("bossfight", distribution_mode)

    def _global_feats(self) -> List[str]:
        return BOSS_FIGHT_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        # Skip attack modes
        data.read_array(elem_size=4)
        feats = [
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
        ]
        # assert len(data.data) == data.offset
        return feats

    def _entity_types(self) -> Dict[int, str]:
        return {
            1: "PlayerBullet",
            2: "Boss",
            3: "Shields",
            4: "EnemyBullet",
            5: "LaserTrail",
            6: "ReflectedBullet",
            7: "Barrier",
            54: "???",
        }
