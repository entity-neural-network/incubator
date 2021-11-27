import numpy as np
import tempfile
from enn_ppo.sample_recorder import SampleRecorder, Sample, Trace


def test_serde_sample() -> None:
    sample = Sample(
        entities={
            "hero": np.array([[1.0, 2.0, 0.3, 100.0, 10.0]]),
            "enemy": np.array(
                [
                    [4.0, -2.0, 0.3, 100.0],
                    [5.0, -2.0, 0.3, 100.0],
                    [6.0, -2.0, 0.3, 100.0],
                ]
            ),
            "box": np.array(
                [
                    [0.0, 0.0, 0.3, 100.0],
                    [1.0, 0.0, 0.3, 100.0],
                    [2.0, 0.0, 0.3, 100.0],
                ]
            ),
        },
        action_masks={
            "move": np.array([0]),
            "shoot": np.array([0]),
            "explode": np.array([4, 5, 6]),
        },
        probabilities={
            "move": np.array([[0.5, 0.2, 0.3, 0.0]]),
            "shoot": np.array([[0.9, 0.1]]),
            "explode": np.array([[0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]),
        },
        reward=0.3124125987123489,
        step=13,
        episode=4213,
    )
    serialized = sample.serialize()
    deserialized = Sample.deserialize(serialized)
    assert deserialized == sample


def test_sampe_recorder() -> None:
    sample = Sample(
        entities={
            "hero": np.array([[1.0, 2.0, 0.3, 100.0, 10.0]]),
            "enemy": np.array(
                [
                    [4.0, -2.0, 0.3, 100.0],
                    [5.0, -2.0, 0.3, 100.0],
                    [6.0, -2.0, 0.3, 100.0],
                ]
            ),
            "box": np.array(
                [
                    [0.0, 0.0, 0.3, 100.0],
                    [1.0, 0.0, 0.3, 100.0],
                    [2.0, 0.0, 0.3, 100.0],
                ]
            ),
        },
        action_masks={
            "move": np.array([0]),
            "shoot": np.array([0]),
            "explode": np.array([4, 5, 6]),
        },
        probabilities={
            "move": np.array([[0.5, 0.2, 0.3, 0.0]]),
            "shoot": np.array([[0.9, 0.1]]),
            "explode": np.array([[0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]),
        },
        reward=0.3124125987123489,
        step=50,
        episode=4213,
    )

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        sample_recorder = SampleRecorder(f.name, act_space=None, obs_space=None)  # type: ignore
        sample_recorder.record(sample)
        # modify the sample
        sample.reward = 1.0
        sample.entities["hero"][0][3] = 200
        sample_recorder.record(sample)
        sample_recorder.close()

        with open(f.name, "rb") as f:
            trace = Trace.deserialize(f.read())
            assert len(trace.samples) == 2
            assert trace.samples[0].reward == 0.3124125987123489
            assert trace.samples[1].reward == 1.0
            assert trace.samples[0].entities["hero"][0][3] == 100
            assert trace.samples[1].entities["hero"][0][3] == 200
