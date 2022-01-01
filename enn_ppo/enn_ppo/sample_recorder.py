from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Tuple

from entity_gym.environment import (
    ActionSpace,
    ObsBatch,
    ObsSpace,
    ragged_buffer_decode,
    ragged_buffer_encode,
)

import tqdm
import numpy as np
import msgpack
import msgpack_numpy


class SampleRecorder:
    """
    Writes samples to disk.
    """

    def __init__(
        self, path: str, act_space: Dict[str, ActionSpace], obs_space: ObsSpace
    ) -> None:
        self.path = path
        self.file = open(path, "wb")
        # TODO: write header and obs space and act space

    def record(self, obs: ObsBatch, step: List[int], episode: List[int]) -> None:
        bytes = msgpack_numpy.dumps(
            {"obs": obs, "step": step, "episode": episode}, default=ragged_buffer_encode
        )
        # Write 8 bytes unsigned int for the size of the serialized sample
        self.file.write(np.uint64(len(bytes)).tobytes())
        self.file.write(bytes)

    def close(self) -> None:
        self.file.close()


@dataclass
class Trace:
    action_space: Dict[str, int]
    obs_space: ObsSpace
    samples: List[Tuple[ObsBatch, List[int], List[int]]]

    @classmethod
    def deserialize(cls, data: bytes, progress_bar: bool = False) -> "Trace":
        samples: List[Tuple[ObsBatch, List[int], List[int]]] = []
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data))

        offset = 0
        while offset < len(data):
            size = int(np.frombuffer(data[offset : offset + 8], dtype=np.uint64)[0])
            offset += 8
            sample = msgpack_numpy.loads(
                data[offset : offset + size],
                object_hook=ragged_buffer_decode,
                strict_map_key=False,
            )
            samples.append((sample["obs"], sample["step"], sample["episode"]))
            offset += size
            if progress_bar:
                pbar.update(size + 8)
        return Trace(None, None, samples)  # type: ignore
