import numpy as np
import torch

from rogue_net.input_norm import InputNorm


def test_correct_normalization() -> None:
    generator = torch.random.manual_seed(0)
    np.random.seed(0)

    size = (1000, 12)
    examples_per_method = 10
    methods = [torch.rand, torch.randn]

    min_batch_size = 16
    max_batch_size = 128

    for method in methods:
        for _ in range(examples_per_method):
            sample = method(size, generator=generator)
            layer = InputNorm(size[1])
            remaining = size[0]
            while remaining > 0:
                # Select a number of samples to take
                batch_size = min(
                    remaining, np.random.randint(min_batch_size, max_batch_size)
                )
                # Take it
                batch = sample[size[0] - remaining : size[0] - remaining + batch_size]
                samples_seen_so_far = sample[: size[0] - remaining + batch_size]
                remaining -= batch_size
                # Compute output
                layer(batch)
                # Compute statistics on the samples seen so far
                mean_so_far = samples_seen_so_far.mean(dim=0)
                std_so_far = samples_seen_so_far.std(dim=0)

                assert layer.count == size[0] - remaining
                # Note that this does not work when atol=1e-7
                assert torch.allclose(layer.mean, mean_so_far, rtol=0, atol=1e-6)
                assert torch.allclose(layer.stddev(), std_so_far, atol=1e-6, rtol=0)


def test_cliprange() -> None:
    generator = torch.random.manual_seed(0)
    np.random.seed(0)

    size = (1000, 12)
    examples_per_method = 10
    methods = [torch.rand, torch.randn]

    min_batch_size = 16
    max_batch_size = 128

    for method in methods:
        for _ in range(examples_per_method):
            sample = method(size, generator=generator)
            cliprange = np.random.uniform() * 7
            layer = InputNorm(size[1], cliprange=cliprange)
            remaining = size[0]
            while remaining > 0:
                # Select a number of samples to take
                batch_size = min(
                    remaining, np.random.randint(min_batch_size, max_batch_size)
                )
                # Take it
                batch = sample[size[0] - remaining : size[0] - remaining + batch_size]
                remaining -= batch_size
                # Compute output
                out = layer(batch)
                assert torch.all(out >= -cliprange)
                assert torch.all(out <= cliprange)


def test_freeze() -> None:
    generator = torch.random.manual_seed(0)
    np.random.seed(0)

    size = (1000, 12)
    examples_per_method = 10
    methods = [torch.rand, torch.randn]

    min_batch_size = 16
    max_batch_size = 128

    for method in methods:
        for _ in range(examples_per_method):
            sample = method(size, generator=generator)
            cliprange = np.random.uniform() * 7
            layer = InputNorm(size[1], cliprange=cliprange)
            remaining = size[0]
            while remaining > 0:
                # Select a number of samples to take
                batch_size = min(
                    remaining, np.random.randint(min_batch_size, max_batch_size)
                )
                # Take it
                batch = sample[size[0] - remaining : size[0] - remaining + batch_size]
                remaining -= batch_size
                # Compute output
                out = layer(batch)
                layer.freeze()
                other = layer(batch)
                assert torch.allclose(other, out, atol=0)
                layer.unfreeze()
