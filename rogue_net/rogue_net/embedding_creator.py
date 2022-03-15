from typing import Dict
from entity_gym.environment import ObsSpace
from torch import nn
import torch

from rogue_net.input_norm import InputNorm


def create_embeddings(obs_space: ObsSpace, d_model: int) -> nn.ModuleDict:
    embeddings: Dict[str, nn.Module] = {}
    for name, entity in obs_space.entities.items():
        if entity.features:
            embeddings[name] = nn.Sequential(
                InputNorm(len(entity.features)),
                nn.Linear(len(entity.features), d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
        else:
            embeddings[name] = FeaturelessEmbedding(d_model)
    return nn.ModuleDict(embeddings)


class FeaturelessEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Parameter(torch.randn(1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.repeat(x.size(0), 1)
