from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping

from entity_gym.environment import ActionSpace, ObsSpace

import tqdm
import numpy as np
import msgpack
import msgpack_numpy


def _numpy_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all(a == b))


@dataclass(eq=False)
class Sample:
    entities: Dict[str, np.ndarray]
    action_masks: Mapping[str, np.ndarray]
    probabilities: Mapping[str, np.ndarray]
    reward: float
    step: int
    episode: int

    def serialize(self) -> bytes:
        return msgpack.packb(asdict(self), default=msgpack_numpy.encode)  # type: ignore

    @classmethod
    def deserialize(cls, data: bytes) -> "Sample":
        return Sample(**msgpack.unpackb(data, object_hook=msgpack_numpy.decode))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sample):
            return False
        if self.entities.keys() != other.entities.keys():
            return False
        for key in self.entities.keys():
            if not _numpy_array_equal(self.entities[key], other.entities[key]):
                return False
        if self.action_masks.keys() != other.action_masks.keys():
            return False
        for key in self.action_masks.keys():
            if not _numpy_array_equal(self.action_masks[key], other.action_masks[key]):
                return False
        if self.probabilities.keys() != other.probabilities.keys():
            return False
        for key in self.probabilities.keys():
            if not _numpy_array_equal(
                self.probabilities[key], other.probabilities[key]
            ):
                return False
        return self.reward == other.reward and self.episode == other.episode


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

    def record(self, sample: Sample) -> None:
        bytes = sample.serialize()
        # Write 8 bytes unsigned int for the size of the serialized sample
        self.file.write(np.uint64(len(bytes)).tobytes())
        self.file.write(bytes)

    def close(self) -> None:
        self.file.close()


@dataclass
class Trace:
    action_space: Dict[str, int]
    obs_space: ObsSpace
    samples: List[Sample]

    @classmethod
    def deserialize(cls, data: bytes, progress_bar: bool = False) -> "Trace":
        samples = []
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data))

        offset = 0
        while offset < len(data):
            size = int(np.frombuffer(data[offset : offset + 8], dtype=np.uint64)[0])
            offset += 8
            sample = Sample.deserialize(data[offset : offset + size])
            samples.append(sample)
            offset += size
            if progress_bar:
                pbar.update(size + 8)
        return Trace(None, None, samples)  # type: ignore
