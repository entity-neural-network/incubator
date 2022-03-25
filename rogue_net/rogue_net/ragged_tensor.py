from dataclasses import dataclass

import torch


@dataclass
class RaggedTensor:
    """
    RaggedTensor represents a 3D tensor with a variable length sequence dimension.

    Example:
    entities in obs 0: [A0, A0, A0, B0]
    entities in obs 1: [A1, B1, B1]
    entities in obs 2: [A2, A2]

    data = [A0, A0, A0, B0, A1, B1, B1, A2, A2]
    batch_index = [0, 0, 0, 0, 1, 1, 1, 2, 2]
    lengths = [4, 3, 2]
    """

    data: torch.Tensor
    """Flattened tensor combining batch and sequence dimension of shape (entities, features)"""
    batch_index: torch.Tensor
    """Gives the batch index of each entity in `data`"""
    lengths: torch.Tensor
    """Gives the number of entities in each observation"""
