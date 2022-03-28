from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer


class StarPilot(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("starpilot", distribution_mode)

    @classmethod
    def _global_feats(cls) -> List[str]:
        return []

    @classmethod
    def deserialize_global_feats(cls, data: ByteBuffer) -> List[float]:
        return []

    @classmethod
    def _entity_types(cls) -> Dict[int, str]:
        return {
            1: "BulletPlayer",
            2: "Bullet2",
            3: "Bullet3",
            4: "Flyer",
            5: "Meteor",
            6: "Cloud",
            7: "Turret",
            8: "FastFlyer",
            9: "FinishLine",
            54: "???",
        }
