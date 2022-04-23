from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer


class StarPilot(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("starpilot", distribution_mode)

    def _global_feats(self) -> List[str]:
        return []

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return []

    def _entity_types(self) -> Dict[int, str]:
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
