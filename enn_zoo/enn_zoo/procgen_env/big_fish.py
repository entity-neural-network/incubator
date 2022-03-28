from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

BIG_FISH_FEATS = ["fish_eaten", "r_inc"]


class BigFish(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("bigfish", distribution_mode)

    @classmethod
    def _global_feats(cls) -> List[str]:
        return BIG_FISH_FEATS

    @classmethod
    def deserialize_global_feats(cls, data: ByteBuffer) -> List[float]:
        return [float(data.read_int()), data.read_float()]

    @classmethod
    def _entity_types(cls) -> Dict[int, str]:
        return {2: "Fish"}
