from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch.nn as nn
import torch

from entity_gym.environment import ObsSpace


@dataclass(frozen=True)
class AbsolutePositionalEncodingConfig:
    d_model: int
    position_extent: List[Tuple[int, int]]
    positional_features: List[str]
    obs_space: ObsSpace

    def __post_init__(self) -> None:
        assert len(self.position_extent) == len(self.positional_features)


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self,
        config: AbsolutePositionalEncodingConfig,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.position_extent = config.position_extent
        self.positional_features = config.positional_features
        # TODO: also need offset if extent doesn't start at 0
        strides = []
        positions = 1
        for start, end in self.position_extent:
            strides.append(float(positions))
            positions *= end - start
        self.positions = positions
        self.strides = torch.tensor(strides).unsqueeze(0)
        # TODO: initialization? minGPT initializes positional embedding as `normal_(mean=0.0, std=0.02)`
        self.embeddings = nn.Embedding(self.positions, self.d_model)
        self.position_feature_indices = {
            entity_name: torch.LongTensor(
                [
                    entity.features.index(feature_name)
                    for feature_name in config.positional_features
                ]
            )
            for entity_name, entity in config.obs_space.entities.items()
        }

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = {}
        for entity_name, features in x.items():
            positions = features[:, self.position_feature_indices[entity_name]]
            indices = torch.tensordot(
                positions,
                # TODO: only send to device once
                self.strides.to(positions.device),
                dims=([1], [1]),
            ).long()
            result[entity_name] = self.embeddings(indices).squeeze(1)
        return result
