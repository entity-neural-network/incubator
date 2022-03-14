from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from dataclasses import dataclass, field
from hyperstate import schema_evolution_cli
from hyperstate.schema.rewrite_rule import (
    ChangeDefault,
    DeleteField,
    RenameField,
    RewriteRule,
)
from hyperstate.schema.versioned import Versioned


class Objective(Enum):
    ALLIED_WEALTH = "ALLIED_WEALTH"
    DISTANCE_TO_CRYSTAL = "DISTANCE_TO_CRYSTAL"
    DISTANCE_TO_ORIGIN = "DISTANCE_TO_ORIGIN"
    DISTANCE_TO_1000_500 = "DISTANCE_TO_1000_500"
    ARENA_TINY = "ARENA_TINY"
    ARENA_TINY_2V2 = "ARENA_TINY_2V2"
    ARENA_MEDIUM = "ARENA_MEDIUM"
    ARENA_MEDIUM_LARGE_MS = "ARENA_MEDIUM_LARGE_MS"
    ARENA = "ARENA"
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"
    SMOL_STANDARD = "SMOL_STANDARD"
    MICRO_PRACTICE = "MICRO_PRACTICE"
    SCOUT = "SCOUT"

    def vs(self) -> bool:
        if (
            self == Objective.ALLIED_WEALTH
            or self == Objective.DISTANCE_TO_CRYSTAL
            or self == Objective.DISTANCE_TO_ORIGIN
            or self == Objective.DISTANCE_TO_1000_500
            or self == Objective.SCOUT
        ):
            return False
        elif (
            self == Objective.ARENA_TINY
            or self == Objective.ARENA_TINY_2V2
            or self == Objective.ARENA_MEDIUM
            or self == Objective.ARENA
            or self == Objective.STANDARD
            or self == Objective.ENHANCED
            or self == Objective.SMOL_STANDARD
            or self == Objective.MICRO_PRACTICE
            or self == Objective.ARENA_MEDIUM_LARGE_MS
        ):
            return True
        else:
            raise Exception(f"Objective.vs not implemented for {self}")

    def naction(self) -> int:
        return 8 + len(self.extra_builds())

    def builds(self) -> List[Tuple[int, int, int, int, int, int]]:
        b = self.extra_builds()
        b.append((0, 1, 0, 0, 0, 0))
        return b

    def extra_builds(self) -> List[Tuple[int, int, int, int, int, int]]:
        # [storageModules, missileBatteries, constructors, engines, shieldGenerators]
        if self == Objective.ARENA:
            return [(1, 0, 1, 0, 0, 0), (0, 2, 0, 0, 0, 0), (0, 1, 0, 0, 1, 0)]
        elif self == Objective.SMOL_STANDARD or self == Objective.STANDARD:
            return [
                (1, 0, 1, 0, 0, 0),
                (0, 2, 0, 0, 0, 0),
                (0, 1, 0, 0, 1, 0),
                (0, 3, 0, 0, 1, 0),
                (0, 2, 0, 0, 2, 0),
                (2, 1, 1, 0, 0, 0),
                (2, 0, 2, 0, 0, 0),
                (2, 0, 1, 1, 0, 0),
                (0, 2, 0, 1, 1, 0),
                (1, 0, 0, 0, 0, 0),
            ]
        elif self == Objective.ENHANCED:
            return [
                # [s, m, c, e, p, l]
                (1, 0, 0, 0, 0, 0),  # 1s
                (1, 0, 1, 0, 0, 0),  # 1s1c
                (0, 1, 0, 0, 1, 0),  # 1m1p
                (0, 0, 0, 0, 0, 2),  # 2l
                (0, 2, 0, 2, 0, 0),  # 2m2e
                (0, 1, 0, 2, 1, 0),  # 1m1p2e
                (0, 2, 0, 1, 1, 0),  # 2m1e1p
                (0, 0, 0, 1, 0, 3),  # 1e3l
                (2, 0, 1, 1, 0, 0),  # 2s1c1e
                (0, 4, 0, 3, 3, 0),  # 4m3e3p
                (0, 0, 0, 4, 1, 5),  # 4e1p5l
            ]
        else:
            return []


@dataclass
class OptimizerConfig:
    # Optimizer ("SGD" or "RMSProp" or "Adam")
    optimizer_type: str = "Adam"
    # Learning rate
    lr: float = 0.0003
    # Momentum
    momentum: float = 0.9
    # Weight decay
    weight_decay: float = 0.0001
    # Batch size during optimization
    batch_size: int = 2048
    # Micro batch size for gradient accumulation
    micro_batch_size: int = 2048
    # Shuffle samples collected during rollout before optimization
    shuffle: bool = True
    # Weighting of value function loss in optimization objective
    vf_coef: float = 1.0
    # Weighting of  entropy bonus in loss function
    entropy_bonus: float = 0.0
    # Maximum gradient norm for gradient clipping
    max_grad_norm: float = 20.0
    # Number of optimizer passes over samples collected during rollout
    epochs: int = 2
    # Exponentially moving averages of model weights
    weights_ema: List[float] = field(default_factory=list)
    # [0.99, 0.997, 0.999, 0.9997, 0.9999]


@dataclass
class PolicyConfig:
    d_agent: int = 256
    d_item: int = 128
    dff_ratio: int = 2
    nhead: int = 8
    item_item_attn_layers: int = 0
    dropout: float = 0.0  # Try 0.1?
    # Construct map of nearby objects populated with scatter connections
    nearby_map: bool = False
    # Width of circles on nearby map
    nm_ring_width = 60
    # Number of rays on nearby map
    nm_nrays = 8
    # Number of rings on nearby map
    nm_nrings = 8
    # Whether to perform convolution on nearby map
    map_conv = False
    # Size of convolution kernel for nearby map
    mc_kernel_size: int = 3
    # Whether the nearby map has 2 channels corresponding to the offset of objects within the tile
    map_embed_offset: bool = False
    # Adds itemwise ff resblock after initial embedding before transformer
    item_ff: bool = True
    # Max number of simultaneously controllable drones
    agents: int = 1
    # Max number of allies observed by each drone
    nally: int = 1
    # Max number of enemies observed by each drone
    nenemy: int = 0
    # Max number of minerals observed by each drone
    nmineral: int = 10
    # Number of map tiles observed by each drone
    ntile: int = 0
    # Number learnable constant valued items observed by each drone
    nconstant: int = 0
    # Use same weights for processing ally and enemy drones
    ally_enemy_same: bool = False
    # Normalization layers ("none", "batchnorm", "layernorm")
    norm: str = "layernorm"
    # Set all initial weights for value function head to zero
    zero_init_vf: bool = True
    # Set initial weights for policy head to small values and biases to zero
    small_init_pi: bool = False


@dataclass
class ObsConfig:
    # Max number of allied drones returned by the env
    allies: int = 10
    # Max number of enemy drones returned by the env
    obs_enemies: int = 10
    # Max number of minerals returned by the env
    obs_minerals: int = 10
    # Max number of map tiles returned by the env
    obs_map_tiles: int = 10
    # Have features for both absolute and relative positions on each object
    obs_keep_abspos: bool = False
    # Whether value function has access to hidden information
    use_privileged: bool = True
    # Global features for width/height of map
    feat_map_size: bool = True
    # Remember last position/time each enemy was seen + missile cooldown feat
    feat_last_seen: bool = False
    # Feature for whether drone is currently visible
    feat_is_visible: bool = True
    # Global features for absolute remaining/elapsed number of timesteps
    feat_abstime: bool = True
    # Feature for whether another drone is currently harvesting a mineral
    feat_mineral_claims: bool = False
    # Harvest action that will freeze drone until one resource has been harvested
    harvest_action: bool = False
    # Pair of actions to disable/enable all build actions
    lock_build_action: bool = False
    # Five features giving distance to closest wall in movement direction, and in movement direction offset by +-pi/2 and +-pi/4
    feat_dist_to_wall: bool = False
    feat_unit_count: bool = True
    feat_construction_progress: bool = True

    # TODO: hack
    feat_rule_msdm = True
    feat_rule_costs = True
    num_builds = 0

    @property
    def drones(self) -> int:
        return self.allies + self.obs_enemies

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
        if self.feat_unit_count:
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
        if self.feat_construction_progress:
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
            + self.obs_minerals * self.mstride()
            + self.obs_map_tiles * self.tstride()
        )

    def endglobals(self) -> int:
        return self.global_features()

    def endallies(self) -> int:
        return self.global_features() + self.dstride() * self.allies

    def endenemies(self) -> int:
        return self.global_features() + self.dstride() * self.drones

    def endmins(self) -> int:
        return self.endenemies() + self.mstride() * self.obs_minerals

    def endtiles(self) -> int:
        return self.endmins() + self.tstride() * self.obs_map_tiles

    def endallenemies(self) -> int:
        return self.endtiles() + self.dstride() * self.enemies()

    def extra_actions(self) -> int:
        if self.lock_build_action:
            return 2
        else:
            return 0

    @property
    def global_drones(self) -> int:
        return self.obs_enemies if self.use_privileged else 0

    @property
    def unit_count(self) -> bool:
        return self.feat_unit_count

    @property
    def construction_progress(self) -> bool:
        return self.feat_construction_progress


@dataclass
class EvalConfig:
    envs: int = 256
    steps: int = 360
    frequency: int = int(1e5)
    model_save_frequency: int = 10
    symmetric: bool = True
    full_eval_frequency: int = 5
    extra_checkpoint_steps: List[int] = field(default_factory=list)


@dataclass
class PPOConfig:
    # Total number of timesteps
    steps: int = int(10e6)
    # Number of environments
    num_envs: int = 64
    # Number of self-play environments (each provides two environments)
    num_self_play: int = 32
    # Number of environments played vs scripted replicator AI
    num_vs_replicator: int = 0
    # Number of environments played vs scripted aggressive replicator AI
    num_vs_aggro_replicator: int = 0
    # Number of environments played vs scripted destroyer AI
    num_vs_destroyer: int = 0
    # Number of sequential steps per rollout
    seq_rosteps: int = 256
    # Discount factor
    gamma: float = 0.99
    # Generalized advantage estimation parameter lambda
    lamb: float = 0.95
    # Normalize advantage values
    norm_advs: bool = True
    # Scaling of reward values
    rewscale: float = 1.0
    # Use PPO-clip instead of vanilla policy gradients objective
    ppo: bool = True
    # PPO cliprange
    cliprange: float = 0.2
    # Use clipped value function objective
    clip_vf: bool = True
    # Split reward evenly amongst all active agents.
    split_reward: bool = False
    # Negative reward applied at each timestep
    liveness_penalty: float = 0.0
    # Extra reward for building a drone type at least once during episode
    build_variety_bonus: float = 0.0
    # Reward received when winning game by eliminating opponent
    win_bonus: float = 0.0
    # Negative reward received when losing game by being eliminated
    loss_penalty: float = 0.0
    # Instantaneous reward received from change in relative amount of resources under allied control
    partial_score: float = 1.0
    # Fraction of shaped reward awarded for minimum health of enemy mothership during episode
    attac: float = 0.0
    # Fraction of shaped reward awarded for maximum health of allied mothership during episode
    protec: float = 0.0
    # Rescale reward values by ema of mean and variance
    rewnorm: bool = False
    rewnorm_emaw: float = 0.97
    max_army_size_score: float = 9999999
    max_enemy_army_size_score: float = 9999999


@dataclass
class TaskConfig:
    objective: Objective = Objective.ARENA_TINY_2V2
    action_delay: int = 0
    use_action_masks: bool = True
    task_hardness: float = 0
    # Max length of games, or default game length for map if 0.
    max_game_length: int = 0
    randomize: bool = True
    # Percentage of maps which are symmetric
    symmetric_map: float = 0.0
    # Linearly increase env symmetry parameter with this slope for every step
    symmetry_increase: float = 2e-8
    # Fraction of maps that use MICRO_PRACTICE instead of the main objective
    mix_mp: float = 0.0
    # Fraction of maps that use randomize ruleset
    rule_rng_fraction: float = 0.0
    # Amount of rule randomization
    rule_rng_amount: float = 1.0
    rule_cost_rng: float = 0.0
    # Automatically adjust environment rules
    adr: bool = False
    # Amount by which task difficulty/map size is increased for each processed frame
    mothership_damage_scale: float = 0.0
    enforce_unit_cap: bool = False
    unit_cap: int = 20

    # Set by ppo
    build_variety_bonus: float = 0.0
    # Set by adr
    cost_variance: float = 0.0


@dataclass
class AdrConfig:
    # Target value for average module cost
    hstepsize: float = 3.0e-6
    average_cost_target: float = 1.0
    cost_variance: float = 0.5
    variety: float = 0.8
    stepsize: float = 0.003
    warmup: int = 100
    initial_hardness: float = 0.0
    # Linearly increase task difficulty/map size
    linear_hardness: bool = False
    # Maximum map area
    max_hardness: float = 150
    # Number of timesteps steps after which hardness starts to increase
    hardness_offset: float = 1e6


@dataclass
class Config(Versioned):
    optimizer: OptimizerConfig
    eval: EvalConfig
    ppo: PPOConfig
    task: TaskConfig
    adr: AdrConfig
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    wandb: bool = True
    trial: Optional[int] = None

    def __post_init__(self) -> None:
        self.obs.feat_rule_msdm = self.task.rule_rng_fraction > 0 or self.task.adr
        self.obs.feat_rule_costs = self.task.rule_cost_rng > 0 or self.task.adr
        self.obs.num_builds = len(self.task.objective.builds())

    @property
    def rosteps(self) -> int:
        return self.ppo.num_envs * self.ppo.seq_rosteps

    def validate(self) -> None:
        assert self.rosteps % self.optimizer.batch_size == 0

    @classmethod
    def version(clz) -> int:
        return 4

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        """
        Returns a list of rewrite rules that can be applied to the given version
        to make it compatible with the next version.
        """
        return {
            0: [
                DeleteField(field=("optimizer", "batches_per_update")),
                DeleteField(field=("optimizer", "bs")),
            ],
            1: [
                RenameField(
                    old_field=("eval", "eval_envs"), new_field=("eval", "envs")
                ),
                RenameField(
                    old_field=("eval", "eval_timesteps"), new_field=("eval", "steps")
                ),
                RenameField(
                    old_field=("eval", "eval_frequency"),
                    new_field=("eval", "frequency"),
                ),
                RenameField(
                    old_field=("eval", "eval_symmetric"),
                    new_field=("eval", "symmetric"),
                ),
            ],
            2: [
                ChangeDefault(
                    field=("task", "mothership_damage_scale"), new_default=0.0
                ),
            ],
            3: [
                DeleteField(field=("version",)),
            ],
        }


if __name__ == "__main__":
    schema_evolution_cli(Config)
