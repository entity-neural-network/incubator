from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math
import time
from enn_zoo.codecraft import rest_client
from enn_zoo.codecraft.rest_client import ObsConfig, Rules
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool
from entity_gym.environment import VecEnv
from entity_gym.environment.vec_env import VecCategoricalActionMask, VecObs
from entity_gym.environment import Environment, ObsSpace
from entity_gym.environment.environment import (
    Action,
    ActionSpace,
    CategoricalActionSpace,
    Entity,
    Observation,
    EpisodeStats,
)

DRONE_FEATS = [
    "x",
    "y",
    "orientation_x",
    "orientation_y",
    "stored_resources",
    "is_constructing",
    "is_harvesting",
    "hitpoints",
    "storage_modules",
    "missile_batteries",
    "constructors",
    "engines",
    "shield_generators",
    "long_range_missiles",
    "is_stunned",
    "is_enemy",
    # feat_last_seen
    # "time_since_visible",
    # "missile_cooldown",
    "long_range_missile_chargeup",
    "is_visible",
    # lock_build_action
    # "build_action_locked",
    # feat_dist_to_wall
    # "distance_to_wall_0",
    # "distance_to_wall_1",
    # "distance_to_wall_2",
    # "distance_to_wall_3",
    # "distance_to_wall_4",
    "available_energy",
    "required_energy",
    # TODO: more than one build
    "constructing_1m",
    # global features
    "relative_elapsed_time",
    "allied_score",
    "map_height",
    "map_width",
    "timestep",
    "remaining_timesteps",
    "msdm",
    "cost_multiplier_1m",
    "unit_count",
]


class CodeCraftEnv(Environment):
    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            entities={
                "ally": Entity(list(DRONE_FEATS)),
                "enemy": Entity(list(DRONE_FEATS)),
                "mineral": Entity(
                    [
                        "x",
                        "y",
                        "size",
                        # "claimed",
                    ]
                ),
                "tile": Entity(
                    [
                        "x",
                        "y",
                        "last_visited_time",
                        "visited",
                    ]
                ),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            # 0-5: turn/movement (4 is no turn, no movement)
            # 6: build [0,1,0,0,0] drone (if minerals > 5)
            # 7: harvest
            "act": CategoricalActionSpace(
                [
                    "move_left",
                    "move_forward",
                    "move_right",
                    "turn_left",
                    "halt",
                    "turn_right",
                    "build_1m",
                    "harvest",
                    # feat_lock_build_action
                    # "unlock_build_action",
                    # "lock_build_action",
                ]
            )
        }

    def reset(self) -> Observation:
        raise NotImplementedError

    def act(self, action: Mapping[str, Action]) -> Observation:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def env_cls(self) -> Type["Environment"]:
        return self.__class__


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
class TaskConfig:
    objective: Objective = Objective.ARENA_TINY_2V2
    use_action_masks: bool = True
    task_hardness: float = 0
    # Max length of games, or default game length for map if 0.
    max_game_length: int = 0
    randomize: bool = True
    # Percentage of maps which are symmetric
    symmetric_map: float = 0.0
    # Linearly increase env symmetry parameter with this slope for every step
    symmetry_increase: float = 2e-8
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


class CodeCraftVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        num_self_play: int,
        objective: Objective = Objective.ALLIED_WEALTH,
        config: TaskConfig = TaskConfig(),
        stagger: bool = True,
        fair: bool = False,
        randomize: bool = False,
        use_action_masks: bool = True,
        obs_config: ObsConfig = ObsConfig(
            allies=1, drones=1, minerals=10, tiles=0, num_builds=1
        ),
        hardness: int = 0,
        symmetric: float = 0.0,
        scripted_opponents: Optional[List[Tuple[str, int]]] = None,
        win_bonus: float = 0.0,
        attac: float = 0.0,
        protec: float = 0.0,
        max_army_size_score: int = 999999,
        max_enemy_army_size_score: int = 999999,
        rule_rng_fraction: float = 0.0,
        rule_rng_amount: float = 0.0,
        rule_cost_rng: float = 0.0,
        max_game_length: Optional[int] = None,
        stagger_offset: float = 0.0,
        loss_penalty: float = 0.0,
        partial_score: float = 1.0,
        create_game_delay: float = 0.0,
    ) -> None:
        assert num_envs >= 2 * num_self_play
        self.num_envs = num_envs
        self.objective = objective
        self.num_self_play = num_self_play
        self.stagger = stagger
        self.stagger_offset = stagger_offset
        self.fair = fair
        self.game_length = 3 * 60 * 60
        self.custom_map: Callable[
            [bool, int, bool], Optional[Dict[str, Any]]
        ] = lambda _1, _2, _3: None
        self.last_map: Optional[Dict[str, Any]] = None
        self.randomize = randomize
        self.use_action_masks = use_action_masks
        self.obs_config = obs_config
        self.hardness = hardness
        self.symmetric = symmetric
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.partial_score = partial_score
        self.attac = attac
        self.protec = protec
        self.max_army_size_score = max_army_size_score
        self.max_enemy_army_size_score = max_enemy_army_size_score
        self.rule_rng_fraction = rule_rng_fraction
        self.rule_rng_amount = rule_rng_amount
        self.rule_cost_rng = rule_cost_rng
        self.rng_ruleset = None
        self.allow_harvesting = objective != Objective.DISTANCE_TO_CRYSTAL
        self.force_harvesting = False
        self.randomize_idle = objective != Objective.ALLIED_WEALTH
        self.config = config
        self.create_game_delay = create_game_delay

        remaining_scripted = num_envs - 2 * num_self_play
        self.scripted_opponents = []
        if scripted_opponents is not None:
            for opponent, count in scripted_opponents:
                remaining_scripted -= count
                for _ in range(count):
                    self.scripted_opponents.append(opponent)
        for _ in range(remaining_scripted):
            self.scripted_opponents.append("idle")
        self.next_opponent_index = 0

        self.builds = objective.extra_builds()
        if objective == Objective.ALLIED_WEALTH:
            self.custom_map = map_allied_wealth
            self.game_length = 1 * 60 * 60
        else:
            raise NotImplementedError(objective)
        if max_game_length is not None:
            self.game_length = max_game_length
        self.build_costs = [sum(modules) for modules in self.builds]
        self.base_naction = 8 + len(self.builds)
        assert len(self.builds) + 1 == obs_config.num_builds

        self.game_count = 0

        self.games: List[Tuple[int, int, str]] = []
        self.eplen: List[int] = []
        self.eprew: List[float] = []
        self.score: List[Optional[float]] = []
        self.performed_builds: List[Any] = []
        self.rulesets: List[Any] = []

    def rules(self) -> Rules:
        # if np.random.uniform(0, 1) < self.rule_rng_fraction:
        #    return random_rules(
        #        2 ** self.config.mothership_damage_scale,
        #        self.rule_cost_rng,
        #        self.rng_ruleset,
        #        self.config.cost_variance,
        #    )
        # else:
        return Rules(
            mothership_damage_multiplier=2 ** self.config.mothership_damage_scale,
            cost_modifiers={build: 1.0 for build in self.objective.builds()},
        )

    def next_opponent(self) -> str:
        opp = self.scripted_opponents[self.next_opponent_index]
        self.next_opponent_index += 1
        if self.next_opponent_index == len(self.scripted_opponents):
            self.next_opponent_index = 0
        return opp

    def reset(self, obs_config: ObsSpace) -> VecObs:
        self.games = []
        self.eplen = []
        self.score = []
        self.performed_builds = []
        self.rulesets = []
        for i in range(self.num_envs - self.num_self_play):
            # spread out initial game lengths to stagger start times
            self_play = i < self.num_self_play
            game_length = (
                int(
                    self.game_length
                    * (i + 1 - self.stagger_offset)
                    // (self.num_envs - self.num_self_play)
                )
                if self.stagger
                else self.game_length
            )
            opponent = "none" if self_play else self.next_opponent()
            ruleset = self.rules()
            game_id = rest_client.create_game(
                game_length,
                0,
                self_play,
                self.next_map(
                    require_default_mothership=opponent not in ["none", "idle"]
                ),
                opponent,
                ruleset,
                self.allow_harvesting,
                self.force_harvesting,
                self.randomize_idle,
            )
            self.game_count += 1

            self.games.append((game_id, 0, opponent))
            self.eplen.append(1)
            self.eprew.append(0)
            self.score.append(None)
            self.performed_builds.append(defaultdict(lambda: 0))
            self.rulesets.append(ruleset)
            if self_play:
                self.games.append((game_id, 1, opponent))
                self.eplen.append(1)
                self.eprew.append(0)
                self.score.append(None)
                self.performed_builds.append(defaultdict(lambda: 0))
                self.rulesets.append(ruleset)

        return self.observe(obs_config)

    def act(
        self, actions: Mapping[str, RaggedBufferI64], obs_filter: ObsSpace
    ) -> VecObs:
        return self.step([list(row) for row in actions["act"].as_array()], obs_filter)

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        # TODO: @cswinter to provide rest API client for grabbing image state from CodeCraft
        # Have to put this here so mypy doesn't complain about abstract instantiation
        raise NotImplementedError()

    def step(
        self,
        actions: List[List[int]],
        # env_subset: Optional[List[int]] = None,
        # obs_config: Optional[ObsConfig] = None,
        obs_filter: ObsSpace,
        action_masks: Optional[Any] = None,
    ) -> VecObs:
        self.step_async(actions, action_masks)
        return self.observe(obs_filter)

    def step_async(
        self,
        actions: List[List[int]],
        # env_subset: Optional[List[int]] = None,
        action_masks: Optional[Any] = None,
    ) -> None:
        game_actions = []
        # games = [self.games[env] for env in env_subset] if env_subset else self.games
        games = self.games
        for (i, ((game_id, player_id, opponent), player_actions)) in enumerate(
            zip(games, actions)
        ):
            if action_masks is not None:
                action_masks_i = action_masks[i]
            player_actions2 = []
            for (drone_index, action) in enumerate(player_actions):
                if action_masks is not None:
                    action_masks_drone = action_masks_i[drone_index]
                # 0-5: turn/movement (4 is no turn, no movement)
                # 6: build [0,1,0,0,0] drone (if minerals > 5)
                # 7: harvest
                move = False
                harvest = False
                turn = 0
                build = []
                lockBuildAction = False
                unlockBuildAction = False
                if action == 0 or action == 1 or action == 2:
                    move = True
                if action == 0 or action == 3:
                    turn = -1
                if action == 2 or action == 5:
                    turn = 1
                if action == 6:
                    build = [(0, 1, 0, 0, 0, 0)]
                if action == 7:
                    harvest = True
                elif action == 8 + len(self.builds) + 1:
                    unlockBuildAction = True
                elif action >= 8:
                    b = action - 8
                    if b < len(self.builds):
                        build = [self.builds[b]]
                    else:
                        build = [(0, 1, 0, 0, 0, 0)]
                if (
                    len(build) > 0
                    and action_masks is not None
                    and action_masks_drone[action] == 1.0
                ):
                    self.performed_builds[i][build[0]] += 1
                player_actions2.append(
                    (move, turn, build, harvest, lockBuildAction, unlockBuildAction)
                )
            game_actions.append((game_id, player_id, player_actions2))

        rest_client.act_batch(game_actions)

    def observe(self, obs_filter: ObsSpace) -> VecObs:
        env_subset: Optional[List[int]] = None
        obs_config: Optional[ObsConfig] = None
        obs_config = obs_config or self.obs_config
        games = [self.games[env] for env in env_subset] if env_subset else self.games
        num_envs = len(games)

        rews = []
        dones = []
        infos = {}
        obs = rest_client.observe_batch_raw(
            obs_config,
            [(gid, pid) for (gid, pid, _) in games],
            allies=obs_config.allies,
            drones=obs_config.drones,
            minerals=obs_config.minerals,
            tiles=obs_config.tiles,
            relative_positions=False,
            v2=True,
            extra_build_actions=self.builds,
            map_size=obs_config.feat_map_size,
            last_seen=obs_config.feat_last_seen,
            is_visible=obs_config.feat_is_visible,
            abstime=obs_config.feat_abstime,
            rule_msdm=obs_config.feat_rule_msdm,
            rule_costs=obs_config.feat_rule_costs,
            enforce_unit_cap=self.config.enforce_unit_cap,
            unit_cap_override=self.config.unit_cap,
        )
        stride = obs_config.stride()
        for i in range(num_envs):
            game = env_subset[i] if env_subset else i
            winner = obs[stride * num_envs + i * obs_config.nonobs_features()]
            outcome = 0
            elimination_win = 0
            if self.objective.vs():
                allied_score = obs[
                    stride * num_envs + i * obs_config.nonobs_features() + 1
                ]
                allied_score = min(allied_score, self.max_army_size_score)
                enemy_score = obs[
                    stride * num_envs + i * obs_config.nonobs_features() + 2
                ]
                enemy_score = min(enemy_score, self.max_enemy_army_size_score)
                min_allied_ms_health = obs[
                    stride * num_envs + i * obs_config.nonobs_features() + 3
                ]
                min_enemy_ms_health = obs[
                    stride * num_envs + i * obs_config.nonobs_features() + 4
                ]
                score = (
                    self.partial_score
                    * 2
                    * allied_score
                    / (allied_score + enemy_score + 1e-8)
                    - 1
                )
                if winner > 0:
                    if enemy_score == 0:
                        score += self.win_bonus
                    else:
                        score -= self.loss_penalty
                if winner > 0:
                    if enemy_score == 0 or allied_score == 0:
                        elimination_win = 1
                    if enemy_score + allied_score == 0:
                        outcome = 0
                    else:
                        outcome = (allied_score - enemy_score) / (
                            enemy_score + allied_score
                        )
                if self.attac > 0:
                    score -= self.attac * min_enemy_ms_health
                if self.protec > 0:
                    score += self.protec * min_allied_ms_health
            elif self.objective == Objective.SCOUT:
                enemy_score = obs[
                    stride * num_envs + i * obs_config.nonobs_features() + 2
                ]
                score = -enemy_score
            elif self.objective == Objective.ALLIED_WEALTH:
                score = (
                    obs[stride * num_envs + i * obs_config.nonobs_features() + 1] * 0.1
                )
            elif self.objective == Objective.DISTANCE_TO_ORIGIN:
                start = stride * i + obs_config.endglobals()
                x = obs[start]
                y = obs[start + 1]
                score = -math.sqrt(x ** 2 + y ** 2) / 1000.0
            elif self.objective == Objective.DISTANCE_TO_CRYSTAL:
                dstart = stride * i + obs_config.endglobals()
                xd = obs[dstart]
                yd = obs[dstart + 1]
                allmstart = stride * i + obs_config.endenemies()
                score = 0.0
                for m in range(obs_config.minerals):
                    mstart = allmstart + obs_config.mstride() * m
                    x = obs[mstart] - xd
                    y = obs[mstart + 1] - yd
                    size = obs[mstart + 2]
                    nearness = 0.5 - math.sqrt(x ** 2 + y ** 2) / 1000.0
                    score = max(score, 0.2 * nearness * size)
            elif self.objective in [Objective.DISTANCE_TO_1000_500]:

                raise Exception(f"Deprecated objective {self.objective}")
            else:
                raise Exception(f"Unknown objective {self.objective}")

            if len(self.builds) > 0:
                max_entropy = math.log(len(self.builds) + 1)
                build_entropy = 0
                s = sum(self.performed_builds[i].values())
                for count in self.performed_builds[i].values():
                    if count > 0:
                        p = count / s
                        build_entropy -= p * math.log(p)
                score += self.config.build_variety_bonus * build_entropy / max_entropy

            if self.score[game] is None:
                self.score[game] = score
            reward = score - self.score[game]
            self.score[game] = score
            self.eprew[game] += reward

            if winner > 0:
                (game_id, pid, opponent_was) = games[i]
                previous_ruleset = self.rulesets[game]
                if pid == 0:
                    self_play = game // 2 < self.num_self_play
                    opponent = "none" if self_play else self.next_opponent()
                    ruleset = self.rules()
                    if self.create_game_delay > 0:
                        time.sleep(self.create_game_delay)
                    game_id = rest_client.create_game(
                        self.game_length,
                        0,
                        self_play,
                        self.next_map(
                            require_default_mothership=opponent not in ["none", "idle"]
                        ),
                        opponent,
                        ruleset,
                        self.allow_harvesting,
                        self.force_harvesting,
                        self.randomize_idle,
                    )
                    self.game_count += 1
                else:
                    game_id, _, opponent = self.games[game - 1]
                    ruleset = self.rulesets[game - 1]
                # print(f"COMPLETED {i} {game} {games[i]} == {self.games[game]} new={game_id}")
                self.games[game] = (game_id, pid, opponent)
                observation = rest_client.observe(game_id, pid)
                # TODO: use actual observation
                if not obs.flags["WRITEABLE"]:
                    obs = obs.copy()
                obs[
                    stride * i : stride * (i + 1)
                ] = 0.0  # codecraft.observation_to_np(observation)

                dones.append(1.0)
                infos[i] = EpisodeStats(self.eplen[game], self.eprew[game])
                # {
                #     "episode": {
                #         "r": self.eprew[game],
                #         "l": self.eplen[game],
                #         "index": game,
                #         "score": self.score[game],
                #         "elimination": elimination_win,
                #         "builds": self.performed_builds[game],
                #         "outcome": outcome,
                #         "opponent": opponent_was,
                #         "ruleset": previous_ruleset,
                #     }
                # }
                self.eplen[game] = 1
                self.eprew[game] = 0
                self.score[game] = None
                self.performed_builds[game] = defaultdict(lambda: 0)
                self.rulesets[game] = ruleset
            else:
                self.eplen[game] += 1
                dones.append(0.0)

            rews.append(reward)

        naction = self.base_naction + obs_config.extra_actions()
        action_mask_elems = naction * obs_config.allies * num_envs
        action_masks = obs[-action_mask_elems:].reshape(-1, obs_config.allies, naction)

        _obs = obs[: stride * self.num_envs].reshape(num_envs, -1)

        globals = _obs[:, : obs_config.endglobals()]

        # filter out all padding
        allies = _obs[:, obs_config.endglobals() : obs_config.endallies()].reshape(
            num_envs, obs_config.allies, obs_config.dstride()
        )
        allies = np.concatenate(
            [allies, globals.reshape(num_envs, 1, obs_config.global_features())], axis=2
        )
        ally_not_padding = allies[:, :, 7] != 0
        # TODO: hack
        ally_not_padding[:, 0] = True
        ragged_allies = RaggedBufferF32.from_flattened(
            allies[ally_not_padding],
            ally_not_padding.sum(axis=1).astype(np.int64),
        )
        enemies = _obs[:, obs_config.endallies() : obs_config.endenemies()].reshape(
            num_envs, obs_config.enemies(), obs_config.dstride()
        )
        not_padding = enemies[:, :, 7] != 0
        ragged_enemies = RaggedBufferF32.from_flattened(
            enemies[not_padding],
            not_padding.sum(axis=1).astype(np.int64),
        )
        minerals = _obs[:, obs_config.endenemies() : obs_config.endmins()].reshape(
            num_envs, obs_config.minerals, obs_config.mstride()
        )
        not_padding = minerals[:, :, 2] != 0
        ragged_minerals = RaggedBufferF32.from_flattened(
            minerals[not_padding],
            not_padding.sum(axis=1).astype(np.int64),
        )
        tiles = _obs[:, obs_config.endmins() : obs_config.endtiles()].reshape(
            num_envs, obs_config.tiles, obs_config.tstride()
        )
        ragged_tiles = RaggedBufferF32.from_array(tiles)

        actors: List[int] = []
        for i in range(num_envs):
            actors.extend(list(range(ragged_allies[i].items())))

        batch = VecObs(
            features={
                "ally": ragged_allies,
                # "enemy": ragged_enemies,
                "mineral": ragged_minerals,
                # "tile": ragged_tiles,
            },
            action_masks={
                "act": VecCategoricalActionMask(
                    actors=RaggedBufferI64.from_flattened(
                        np.array(actors, dtype=np.int64).reshape(-1, 1),
                        ragged_allies.size1(),
                    ),
                    mask=RaggedBufferBool.from_flattened(
                        action_masks[ally_not_padding] == 1.0, ragged_allies.size1()
                    ),
                ),
            },
            reward=np.array(rews),
            done=np.array(dones) == 1.0,
            end_of_episode_info=infos,
        )
        return batch

    def close(self) -> None:
        # Run all games to completion
        done: Dict[int, bool] = defaultdict(lambda: False)
        running = len(self.games)
        while running > 0:
            game_actions: List[Any] = []
            active_games = []
            for (game_id, player_id, _) in self.games:
                if not done[game_id]:
                    active_games.append((game_id, player_id))
                    game_actions.append(
                        (game_id, player_id, [(False, 0, [], False, False, False)])
                    )
            rest_client.act_batch(game_actions)
            obs = rest_client.observe_batch(active_games)
            for o, (game_id, _) in zip(obs, active_games):
                if o["winner"]:
                    done[game_id] = True
                    running -= 1

    def next_map(
        self, require_default_mothership: bool = False
    ) -> Optional[Dict[str, Any]]:
        if self.fair:
            map = self.fair_map(require_default_mothership)
        else:
            map = self.custom_map(
                self.randomize, self.hardness, require_default_mothership
            )
        if map:
            map["symmetric"] = np.random.rand() < self.symmetric
        return map

    def fair_map(
        self, require_default_mothership: bool = False
    ) -> Optional[Dict[str, Any]]:
        if self.last_map is None:
            self.last_map = self.custom_map(
                self.randomize, self.hardness, require_default_mothership
            )
            assert self.last_map is not None
            return self.last_map
        else:
            result = self.last_map
            self.last_map = None
            p1 = result["player1Drones"]
            result["player1Drones"] = result["player2Drones"]
            result["player2Drones"] = p1
            return result

    def __len__(self) -> int:
        return self.num_envs

    def env_cls(cls) -> Type[Environment]:
        return CodeCraftEnv


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)  # type: ignore


def dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


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
