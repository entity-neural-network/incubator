from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

BIG_FISH_FEATS = ["fish_eaten", "r_inc"]


class BigFish(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("bigfish", distribution_mode)

    def _global_feats(self) -> List[str]:
        return BIG_FISH_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [float(data.read_int()), data.read_float()]

    def _entity_types(self) -> Dict[int, str]:
        return {2: "Fish"}
