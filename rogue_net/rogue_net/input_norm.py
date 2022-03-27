from typing import Optional

import torch
from torch import nn


class InputNorm(nn.Module):
    """
    Computes a running mean/variance of input features and performs normalization.
    Adapted from https://www.johndcook.com/blog/standard_deviation/
    """

    # Pretend that `count` is a float to make MyPy happy
    count: float

    def __init__(self, num_features: int, cliprange: float = 5) -> None:
        super().__init__()

        self.cliprange = cliprange
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("squares_sum", torch.zeros(num_features))
        self.fp16 = False
        self._stddev: Optional[torch.Tensor] = None
        self._dirty = True
        self._frozen = False

    def freeze(self) -> None:
        """
        Freeze the running statistics, thus the normalization.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """
        Unfreeze the running statistics, thus the normalization.
        """
        self._frozen = False

    def update(self, input: torch.Tensor) -> None:
        self._dirty = True
        count = input.numel() // input.size(-1)
        if count == 0:
            return
        dreduce = tuple(range(0, input.dim() - 1))
        mean = input.mean(dim=dreduce)
        square_sum = ((input - mean) * (input - mean)).sum(dim=dreduce)
        if self.count == 0:
            self.count += count
            self.mean = mean
            self.squares_sum = square_sum
        else:
            # This does not follow directly Welford's method since it is a batched update
            # Instead we consider computing the statistics of two sets, A="current set so far" B="current batch"
            # See Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979), "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.", Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
            delta = mean - self.mean
            self.mean += delta * count / (count + self.count)
            self.squares_sum += square_sum + torch.square(
                delta
            ) * count * self.count / (count + self.count)
            self.count += count

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.training and not self._frozen:
                self.update(input)
            if self.count > 1:
                input = (input - self.mean) / self.stddev()
            input = torch.clamp(input, -self.cliprange, self.cliprange)

        return input.half() if self.fp16 else input

    def enable_fp16(self) -> None:
        # Convert buffers back to fp32, fp16 has insufficient precision and runs into overflow on squares_sum
        self.float()
        self.fp16 = True

    def stddev(self) -> torch.Tensor:
        if self._dirty or self._stddev is None:
            sd = torch.sqrt(self.squares_sum / (self.count - 1))
            sd[sd == 0] = 1
            self._stddev = sd
            self._dirty = False
        return self._stddev
