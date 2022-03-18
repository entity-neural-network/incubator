from typing import Dict, List, Mapping, Tuple
from enn_zoo.procgen_env.deserializer import ByteBuffer, ProcgenState
from entity_gym.environment import *
from procgen import ProcgenGym3Env


ENTITY_FEATS = [
    "x",
    "y",
    "vx",
    "vy",
    "rx",
    "ry",
    "type",
    "image_type",
    "image_theme",
    "render_z",
    "will_erase",
    "collides_with_entities",
    "collision_margin",
    "rotation",
    "vrot",
    "is_reflected",
    "fire_time",
    "spawn_time",
    "life_time",
    "expire_time",
    "use_abs_coords",
    "friction",
    "smart_step",
    "avoids_collisions",
    "auto_erase",
    "alpha",
    "health",
    "theta",
    "grow_rate",
    "alpha_decay",
    "climber_spawn_x",
]

BIG_FISH_FEATS = ["fish_eaten", "r_inc"]
EXTRA_FEATS = ["dsize", "bigger"]


class BigFish(Environment):
    def __init__(self, extra_feats: bool = False) -> None:
        self.env = ProcgenGym3Env(
            num=1,
            env_name="bigfish",
            start_level=0,
            num_levels=1000000,
        )
        self.total_reward = 0.0
        self.step = 0
        self.extra_feats = extra_feats

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Player": Entity(features=ENTITY_FEATS + BIG_FISH_FEATS),
                "Fish": Entity(features=ENTITY_FEATS + BIG_FISH_FEATS + EXTRA_FEATS),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[ActionType, ActionSpace]:
        return {
            "act": CategoricalActionSpace(
                [
                    "left-down",
                    "left",
                    "left-up",
                    "down",
                    "none",
                    "up",
                    "right-down",
                    "right",
                    "right-up",
                    "d",
                    "a",
                    "w",
                    "s",
                    "q",
                    "e",
                ]
            )
        }

    def reset(self) -> Observation:
        self.total_reward = 0.0
        self.step = 0
        return self.observe()

    def observe(self) -> Observation:
        states = self.env.callmethod("get_state")
        state = ProcgenState.from_bytes(ByteBuffer(states[0]))

        player = []
        fish = []
        global_feats = [state.fish_eaten, state.r_inc]
        player_entity = [e for e in state.entities if e.type == 0][0]
        for entity in state.entities:
            if entity.type == 0:
                player.append(entity.to_list() + global_feats)
            elif entity.type == 2:
                if self.extra_feats:
                    extra_feats = [
                        entity.rx - player_entity.rx,
                        0.0 if entity.rx > player_entity.rx else 1.0,
                    ]
                else:
                    extra_feats = [0.0, 0.0]
                fish.append(entity.to_list() + global_feats + extra_feats)
            else:
                print("warning: unknown entity type", entity.type)
                __import__("ipdb").set_trace()

        return Observation.from_entity_obs(
            entities={
                "Player": EntityObs(features=player, ids=[0]),
                "Fish": EntityObs(features=fish),
            },
            actions={"act": CategoricalActionMask(actor_types=["Player"])},
            done=state.game.step_data.done == 1,
            reward=state.game.step_data.reward,
            end_of_episode_info=EpisodeStats(
                length=self.step,
                total_reward=self.total_reward,
            )
            if state.game.step_data.done == 1
            else None,
        )

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        act = actions["act"]
        assert isinstance(act, CategoricalAction)
        self.env.act(act.actions)
        self.step += 1
        obs = self.observe()
        self.total_reward += obs.reward
        return obs
