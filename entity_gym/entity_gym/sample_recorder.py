from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Type

from entity_gym.environment import (
    Action,
    ActionSpace,
    Environment,
    ObsBatch,
    ObsSpace,
    VecEnv,
    ragged_buffer_decode,
    ragged_buffer_encode,
)
from ragged_buffer import RaggedBufferF32, RaggedBufferI64

import tqdm
import numpy as np
import msgpack_numpy


@dataclass
class Sample:
    obs: ObsBatch
    step: List[int]
    episode: List[int]
    actions: Sequence[Mapping[str, Action]]
    probs: Dict[str, RaggedBufferF32]
    logits: Optional[Dict[str, RaggedBufferF32]]

    def serialize(self) -> bytes:
        return msgpack_numpy.dumps(  # type: ignore
            {
                "obs": self.obs,
                "step": self.step,
                "episode": self.episode,
                "actions": self.actions,
                "probs": self.probs,
                "logits": self.logits,
            },
            default=ragged_buffer_encode,
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "Sample":
        return Sample(
            **msgpack_numpy.loads(
                data, object_hook=ragged_buffer_decode, strict_map_key=False
            )
        )


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

    def record(
        self,
        sample: Sample,
    ) -> None:
        bytes = sample.serialize()
        # Write 8 bytes unsigned int for the size of the serialized sample
        self.file.write(np.uint64(len(bytes)).tobytes())
        self.file.write(bytes)

    def close(self) -> None:
        self.file.close()


class SampleRecordingVecEnv(VecEnv):
    def __init__(self, inner: VecEnv, out_path: str) -> None:
        self.inner = inner
        self.out_path = out_path
        self.sample_recorder = SampleRecorder(
            out_path, inner.env_cls().action_space(), inner.env_cls().obs_space()
        )
        self.last_obs: Optional[ObsBatch] = None
        self.episodes = list(range(len(inner)))
        self.curr_step = [0] * len(inner)
        self.next_episode = len(inner)

    def reset(self, obs_config: ObsSpace) -> ObsBatch:
        self.curr_step = [0] * len(self)
        self.last_obs = self.record_obs(self.inner.reset(obs_config))
        return self.last_obs

    def record_obs(self, obs: ObsBatch) -> ObsBatch:
        for i, done in enumerate(obs.done):
            if done:
                self.episodes[i] = self.next_episode
                self.next_episode += 1
                self.curr_step[i] = 0
            else:
                self.curr_step[i] += 1
        self.last_obs = obs
        return obs

    def act(
        self,
        actions: Sequence[Mapping[str, Action]],
        obs_filter: ObsSpace,
        probs: Optional[Dict[str, RaggedBufferF32]] = None,
        logits: Optional[Dict[str, RaggedBufferF32]] = None,
    ) -> ObsBatch:
        if probs is None:
            probs = {}
        # with tracer.span("record_samples"):
        assert self.last_obs is not None
        self.sample_recorder.record(
            Sample(
                self.last_obs,
                step=list(self.curr_step),
                episode=list(self.episodes),
                actions=actions,
                probs=probs,
                logits=logits,
            )
        )
        return self.record_obs(self.inner.act(actions, obs_filter))

    def env_cls(cls) -> Type[Environment]:
        return super().env_cls()

    def __len__(self) -> int:
        return len(self.inner)

    def close(self) -> None:
        self.sample_recorder.close()
        print("Recorded samples to: ", self.sample_recorder.path)
        self.inner.close()


@dataclass
class Trace:
    action_space: Dict[str, int]
    obs_space: ObsSpace
    samples: List[Sample]

    @classmethod
    def deserialize(cls, data: bytes, progress_bar: bool = False) -> "Trace":
        samples: List[Sample] = []
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data))

        offset = 0
        while offset < len(data):
            size = int(np.frombuffer(data[offset : offset + 8], dtype=np.uint64)[0])
            offset += 8
            samples.append(Sample.deserialize(data[offset : offset + size]))
            offset += size
            if progress_bar:
                pbar.update(size + 8)
        return Trace(None, None, samples)  # type: ignore
