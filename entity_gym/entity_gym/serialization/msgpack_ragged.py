from typing import Any
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool
import msgpack_numpy

from entity_gym.environment.environment import *
from entity_gym.environment.vec_env import *


# For security reasons we don't want to deserialize classes that are not in this list.
WHITELIST = {
    "ObsSpace": ObsSpace,
    "VecObs": VecObs,
    "VecCategoricalActionMask": VecCategoricalActionMask,
    "VecSelectEntityActionMask": VecSelectEntityActionMask,
    "SelectEntityAction": SelectEntityAction,
    "CategoricalAction": CategoricalAction,
    "Entity": Entity,
    "EpisodeStats": EpisodeStats,
}


def ragged_buffer_encode(obj: Any) -> Any:
    if isinstance(obj, RaggedBufferF32) or isinstance(obj, RaggedBufferI64) or isinstance(obj, RaggedBufferBool):  # type: ignore
        flattened = obj.as_array()
        lengths = obj.size1()
        return {
            "__flattened__": msgpack_numpy.encode(flattened),
            "__lengths__": msgpack_numpy.encode(lengths),
        }
    elif hasattr(obj, "__dict__"):
        return {"__classname__": obj.__class__.__name__, "data": vars(obj)}
    else:
        return obj


def ragged_buffer_decode(obj: Any) -> Any:
    if "__flattened__" in obj:
        flattened = msgpack_numpy.decode(obj["__flattened__"])
        lengths = msgpack_numpy.decode(obj["__lengths__"])

        dtype = flattened.dtype

        if dtype == np.float32:
            return RaggedBufferF32.from_flattened(flattened, lengths)
        elif dtype == int:
            return RaggedBufferI64.from_flattened(flattened, lengths)
    elif "__classname__" in obj:
        classname = obj["__classname__"]
        if classname in WHITELIST:
            cls_name = globals()[classname]
            return cls_name(**obj["data"])
        else:
            raise RuntimeError(
                f"Attempt to deserialize class {classname} outside whitelist."
            )
    else:
        return obj
