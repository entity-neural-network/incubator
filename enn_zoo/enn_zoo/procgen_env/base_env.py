from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Mapping
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


class BaseEnv(Environment):
    def __init__(self, env_name: str, distribution_mode: str) -> None:
        self.env = ProcgenGym3Env(
            num=1,
            env_name=env_name,
            start_level=0,
            num_levels=0,
            distribution_mode=distribution_mode,
        )

    @classmethod
    @abstractmethod
    def _global_feats(cls) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def deserialize_global_feats(cls, data: ByteBuffer) -> List[float]:
        pass

    @classmethod
    @abstractmethod
    def _entity_types(cls) -> Dict[int, str]:
        pass

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Player": Entity(features=ENTITY_FEATS + cls._global_feats()),
                **{
                    entity_type: Entity(features=ENTITY_FEATS + cls._global_feats())
                    for entity_type in cls._entity_types().values()
                },
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
        return self.observe()

    def observe(self) -> Observation:
        states = self.env.callmethod("get_state")
        data = ByteBuffer(states[0])
        state = ProcgenState.from_bytes(data)

        entities = defaultdict(list)
        entity_types = self._entity_types()
        global_feats = self.__class__.deserialize_global_feats(data)
        for entity in state.entities:
            if entity.type == 0:
                entities["Player"].append(entity.to_list() + global_feats)
            elif entity.type in entity_types:
                entities[entity_types[entity.type]].append(
                    entity.to_list() + global_feats
                )
            else:
                print("warning: unknown entity type", entity.type)
                __import__("ipdb").set_trace()

        return Observation.from_entity_obs(
            entities={
                name: EntityObs(
                    features=features, ids=[0] if name == "Player" else None
                )
                for name, features in entities.items()
            },
            actions={"act": CategoricalActionMask(actor_types=["Player"])},
            done=state.game.step_data.done == 1,
            reward=state.game.step_data.reward,
        )

    def act(self, actions: Mapping[ActionType, Action]) -> Observation:
        act = actions["act"]
        assert isinstance(act, CategoricalAction)
        self.env.act(act.actions)
        return self.observe()
