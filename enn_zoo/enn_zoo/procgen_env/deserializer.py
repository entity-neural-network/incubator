import struct
from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound="ProcgenState")


class ByteBuffer:
    def __init__(self, data: bytes) -> None:
        self.offset = 0
        self.data = data

    def read_int(self) -> int:
        return struct.unpack("i", self.read(4))[0]  # type: ignore

    def read_float(self) -> float:
        return struct.unpack("f", self.read(4))[0]  # type: ignore

    def read_array(self, elem_size: int) -> bytes:
        return self.read(elem_size * self.read_int())

    def read_float_array(self) -> npt.NDArray[np.float32]:
        return np.frombuffer(self.read_array(elem_size=4), dtype=np.float32)  # type: ignore

    def read_int_array(self) -> npt.NDArray[np.int32]:
        return np.frombuffer(self.read_array(elem_size=4), dtype=np.int32)  # type: ignore

    def read_str(self) -> str:
        return self.read_array(elem_size=1).decode("utf-8")

    def read(self, n: int) -> bytes:
        self.offset += n
        return self.data[self.offset - n : self.offset]


@dataclass
class ProcgenOpts:
    paint_vel_info: int
    use_generated_assets: int
    use_monochrome_assets: int
    restrict_themes: int
    use_backgrounds: int
    center_agent: int
    debug_mode: int
    distribution_mode: int
    use_sequential_levels: int

    use_easy_jump: int
    plain_assets: int
    physics_mode: int

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "ProcgenOpts":
        return cls(
            paint_vel_info=data.read_int(),
            use_generated_assets=data.read_int(),
            use_monochrome_assets=data.read_int(),
            restrict_themes=data.read_int(),
            use_backgrounds=data.read_int(),
            center_agent=data.read_int(),
            debug_mode=data.read_int(),
            distribution_mode=data.read_int(),
            use_sequential_levels=data.read_int(),
            use_easy_jump=data.read_int(),
            plain_assets=data.read_int(),
            physics_mode=data.read_int(),
        )


@dataclass
class RngState:
    is_seeded: int
    state: bytes

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "RngState":
        return cls(
            is_seeded=data.read_int(),
            # Don't care about state and it's very large so truncate to 1 byte
            state=data.read_array(elem_size=1)[0:1],
        )


@dataclass
class StepData:
    reward: float
    done: int
    level_complete: int

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "StepData":
        return cls(
            reward=data.read_float(),
            done=data.read_int(),
            level_complete=data.read_int(),
        )


@dataclass
class ProcgenGame:
    version: int
    name: str
    opts: ProcgenOpts
    grid_step: int
    level_seed_low: int
    level_seed_high: int
    game_type: int
    game_n: int
    level_seed_rand_gen: RngState
    rand_gen: RngState
    step_data: StepData
    action: int
    timeout: int
    current_level_seed: int
    prev_level_seed: int
    episodes_remaining: int
    episode_done: int
    last_reward_timer: int
    last_reward: float
    default_action: int
    fixed_asset_seed: int
    cur_time: int
    is_waiting_for_step: int

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "ProcgenGame":
        return cls(
            version=data.read_int(),
            name=data.read_str(),
            opts=ProcgenOpts.from_bytes(data),
            grid_step=data.read_int(),
            level_seed_low=data.read_int(),
            level_seed_high=data.read_int(),
            game_type=data.read_int(),
            game_n=data.read_int(),
            level_seed_rand_gen=RngState.from_bytes(data),
            rand_gen=RngState.from_bytes(data),
            step_data=StepData.from_bytes(data),
            action=data.read_int(),
            timeout=data.read_int(),
            current_level_seed=data.read_int(),
            prev_level_seed=data.read_int(),
            episodes_remaining=data.read_int(),
            episode_done=data.read_int(),
            last_reward_timer=data.read_int(),
            last_reward=data.read_float(),
            default_action=data.read_int(),
            fixed_asset_seed=data.read_int(),
            cur_time=data.read_int(),
            is_waiting_for_step=data.read_int(),
        )

    @classmethod
    def just_step_data(cls, data: ByteBuffer) -> StepData:
        # Version
        data.offset += 4
        # Name
        data.read_array(elem_size=1)
        # Procgen opts
        data.read(4 * 12)
        # grid_step, level_seed_low, level_seed_high, game_type, game_n
        data.offset += 4 * 5
        # level_seed_rand_gen, rand_gen
        data.offset += 4
        data.read_array(elem_size=1)
        data.offset += 4
        data.read_array(elem_size=1)
        step_data = StepData.from_bytes(data)
        # action, timeout, current_level_seed, prev_level_seed, episodes_remaining, episode_done, last_reward_timer, last_reward, default_action, fixed_asset_seed, cur_time, is_waiting_for_step
        data.offset += 4 * 12
        return step_data


@dataclass
class Entity:
    x: float
    y: float
    vx: float
    vy: float
    rx: float
    ry: float
    type: int
    image_type: int
    image_theme: int
    render_z: int
    will_erase: int
    collides_with_entities: int
    collision_margin: float
    rotation: float
    vrot: float
    is_reflected: int
    fire_time: int
    spawn_time: int
    life_time: int
    expire_time: int
    use_abs_coords: int
    friction: float
    smart_step: int
    avoids_collisions: int
    auto_erase: int
    alpha: float
    health: float
    theta: float
    grow_rate: float
    alpha_decay: float
    climber_spawn_x: float

    @classmethod
    def list_from_bytes(cls, data: ByteBuffer) -> List["Entity"]:
        return [cls.from_bytes(data) for _ in range(data.read_int())]

    @classmethod
    def array_from_bytes(cls, data: ByteBuffer) -> npt.NDArray[np.float32]:
        count = data.read_int()
        array = np.empty(shape=(count, 31), dtype=np.float32)
        for i in range(count):
            array[i, 0:6] = np.frombuffer(data.read(6 * 4), dtype=np.float32)
            array[i, 6:12] = np.frombuffer(data.read(6 * 4), dtype=np.int32).astype(
                np.float32
            )
            array[i, 12:15] = np.frombuffer(data.read(3 * 4), dtype=np.float32)
            array[i, 15:21] = np.frombuffer(data.read(6 * 4), dtype=np.int32).astype(
                np.float32
            )
            array[i, 21] = data.read_float()
            array[i, 22:25] = np.frombuffer(data.read(3 * 4), dtype=np.int32).astype(
                np.float32
            )
            array[i, 25:31] = np.frombuffer(data.read(6 * 4), dtype=np.float32)
        return array

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "Entity":
        return cls(
            x=data.read_float(),
            y=data.read_float(),
            vx=data.read_float(),
            vy=data.read_float(),
            rx=data.read_float(),
            ry=data.read_float(),
            type=data.read_int(),
            image_type=data.read_int(),
            image_theme=data.read_int(),
            render_z=data.read_int(),
            will_erase=data.read_int(),
            collides_with_entities=data.read_int(),
            collision_margin=data.read_float(),
            rotation=data.read_float(),
            vrot=data.read_float(),
            is_reflected=data.read_int(),
            fire_time=data.read_int(),
            spawn_time=data.read_int(),
            life_time=data.read_int(),
            expire_time=data.read_int(),
            use_abs_coords=data.read_int(),
            friction=data.read_float(),
            smart_step=data.read_int(),
            avoids_collisions=data.read_int(),
            auto_erase=data.read_int(),
            alpha=data.read_float(),
            health=data.read_float(),
            theta=data.read_float(),
            grow_rate=data.read_float(),
            alpha_decay=data.read_float(),
            climber_spawn_x=data.read_float(),
        )

    def to_list(self) -> List[float]:
        return [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.rx,
            self.ry,
            self.type,
            self.image_type,
            self.image_theme,
            self.render_z,
            self.will_erase,
            self.collides_with_entities,
            self.collision_margin,
            self.rotation,
            self.vrot,
            self.is_reflected,
            self.fire_time,
            self.spawn_time,
            self.life_time,
            self.expire_time,
            self.use_abs_coords,
            self.friction,
            self.smart_step,
            self.avoids_collisions,
            self.auto_erase,
            self.alpha,
            self.health,
            self.theta,
            self.grow_rate,
            self.alpha_decay,
            self.climber_spawn_x,
        ]


@dataclass
class Grid:
    w: int
    h: int
    data: List[int]

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "Grid":
        w = data.read_int()
        h = data.read_int()
        size = data.read_int()
        assert size == w * h
        return cls(w, h, [data.read_int() for _ in range(w * h)])


@dataclass
class ProcgenState:
    game: ProcgenGame
    grid_size: int
    entities: List[Entity]
    use_procgen_background: int
    background_index: int
    bg_tile_ratio: float
    bg_pct_x: float
    char_dim: float
    last_move_action: int
    move_action: int
    special_action: int
    mixrate: float
    maxspeed: float
    max_jump: float
    action_vx: float
    action_vy: float
    action_vrot: float
    center_x: float
    center_y: float
    random_agent_start: int
    has_useful_vel_info: int
    step_rand_int: int
    asset_rand_gen: RngState
    main_width: int
    main_height: int
    out_of_bounds_object: int
    unit: float
    view_dim: float
    x_off: float
    y_off: float
    visibility: float
    min_visibility: float
    grid: Grid

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "ProcgenState":
        return cls(
            game=ProcgenGame.from_bytes(data),
            grid_size=data.read_int(),
            entities=Entity.list_from_bytes(data),
            use_procgen_background=data.read_int(),
            background_index=data.read_int(),
            bg_tile_ratio=data.read_float(),
            bg_pct_x=data.read_float(),
            char_dim=data.read_float(),
            last_move_action=data.read_int(),
            move_action=data.read_int(),
            special_action=data.read_int(),
            mixrate=data.read_float(),
            maxspeed=data.read_float(),
            max_jump=data.read_float(),
            action_vx=data.read_float(),
            action_vy=data.read_float(),
            action_vrot=data.read_float(),
            center_x=data.read_float(),
            center_y=data.read_float(),
            random_agent_start=data.read_int(),
            has_useful_vel_info=data.read_int(),
            step_rand_int=data.read_int(),
            asset_rand_gen=RngState.from_bytes(data),
            main_width=data.read_int(),
            main_height=data.read_int(),
            out_of_bounds_object=data.read_int(),
            unit=data.read_float(),
            view_dim=data.read_float(),
            x_off=data.read_float(),
            y_off=data.read_float(),
            visibility=data.read_float(),
            min_visibility=data.read_float(),
            grid=Grid.from_bytes(data),
        )


def print_coinrun_grid(grid: Grid) -> None:

    # const int GOAL = 1;
    # const int SAW = 2;
    # const int SAW2 = 3;
    # const int ENEMY = 5;
    # const int ENEMY1 = 6;
    # const int ENEMY2 = 7;

    # const int PLAYER_JUMP = 9;
    # const int PLAYER_RIGHT1 = 12;
    # const int PLAYER_RIGHT2 = 13;

    # const int WALL_MID = 15;
    # const int WALL_TOP = 16;
    # const int LAVA_MID = 17;
    # const int LAVA_TOP = 18;
    # const int ENEMY_BARRIER = 19;

    # const int CRATE = 20;
    SYMBOLS = {
        1: "G",
        2: "S",
        3: "S",
        15: "W",
        16: "W",
        17: "L",
        18: "L",
        19: "B",
        100: " ",
        4096: "x",
    }
    for y in reversed(range(grid.h)):
        for x in range(grid.w):
            print(
                SYMBOLS.get(grid.data[y * grid.w + x], str(grid.data[y * grid.w + x])),
                end="",
            )
        print()


if __name__ == "__main__":
    import random

    from procgen import ProcgenGym3Env

    env = ProcgenGym3Env(
        num=1, env_name="bigfish", start_level=random.randint(0, 1000), num_levels=1000
    )
    states = env.callmethod("get_state")

    state = ProcgenState.from_bytes(ByteBuffer(states[0]))
    print(state)
    print_coinrun_grid(state.grid)
    for entity in state.entities:
        print(entity)
