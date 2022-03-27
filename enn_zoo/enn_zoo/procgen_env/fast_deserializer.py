from typing import List, TypeVar, Type
import struct
from dataclasses import dataclass

from enn_zoo.procgen_env.deserializer import ByteBuffer, Entity, ProcgenGame, StepData


@dataclass
class MinimalProcgenState:
    step_data: StepData
    grid_size: int
    entities: List[Entity]

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "MinimalProcgenState":
        step_data = ProcgenGame.just_step_data(data)
        grid_size = data.read_int()
        entities = Entity.list_from_bytes(data)
        # use_procgen_background, background_index, bg_tile_ratio, bg_pct_x, char_dim, last_move_action, move_action, special_action, mixrate, maxspeed, max_jump, action_vx, action_vy, action_vrot, center_x, center_y, random_agent_start, has_useful_vel_info, step_rand_int
        data.offset += 4 * 19
        # asset_rand_gen
        data.offset += 4
        data.read_array()
        # main_width, main_height, out_of_bounds_object, unit, view_dim, x_off, y_off, visibility, min_visibility
        data.offset += 4 * 9
        # grid
        w = data.read_int()
        h = data.read_int()
        data.offset += 4 * w * h
        return cls(step_data, grid_size, entities)
