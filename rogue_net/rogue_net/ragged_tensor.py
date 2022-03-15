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

    data = [A0, A0, A0, A1, A2, A2, B0, B1, B1]
    index_map = [0, 1, 2, 6, 3, 7, 8, 4, 5]
    batch_index = [0, 0, 0, 1, 2, 2, 0, 1, 1]
    lengths = [4, 3, 2]
    """

    data: torch.Tensor
    """Flattened tensor combining batch and sequence dimension of shape (entities, features)"""
    index_map: torch.Tensor
    """Maps the index of entities sorted first by batch index then by entity type to their index in `data`"""
    batch_index: torch.Tensor
    """Gives the batch index of each entity in `data`"""
    lengths: torch.Tensor
    """Gives the number of entities in each observation"""
