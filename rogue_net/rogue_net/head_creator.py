from typing import Dict
from torch import nn
import torch
import numpy as np
from entity_gym.environment import ActionSpace, CategoricalActionSpace
from typing import Dict


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)  # type: ignore
    return layer


def create_head_for(space: ActionSpace, d_model: int) -> nn.Module:
    if isinstance(space, CategoricalActionSpace):
        return layer_init(nn.Linear(d_model, len(space.choices)), std=0.01)
    raise NotImplementedError()


def create_value_head(d_model: int) -> nn.Module:
    value_head = nn.Linear(d_model, 1)
    value_head.weight.data.fill_(0.0)
    value_head.bias.data.fill_(0.0)
    return value_head


def create_action_heads(
    action_space: Dict[str, ActionSpace], d_model: int
) -> nn.ModuleDict:
    action_heads = {}
    for name, space in action_space.items():
        action_heads[name] = create_head_for(space, d_model)
    return nn.ModuleDict(action_heads)
