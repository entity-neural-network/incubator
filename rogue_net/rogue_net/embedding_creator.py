from entity_gym.environment import ObsSpace
from torch import nn

from rogue_net.input_norm import InputNorm


def create_embeddings(obs_space: ObsSpace, d_model: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            name: nn.Sequential(
                InputNorm(len(entity.features)),
                nn.Linear(len(entity.features), d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
            for name, entity in obs_space.entities.items()
        }
    )
