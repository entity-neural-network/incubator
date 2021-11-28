from typing import Dict
from torch import nn

from entity_gym.environment import ActionSpace, CategoricalActionSpace


def create_head_for(space: ActionSpace) -> nn.Module:
    if isinstance(space, CategoricalActionSpace):
        return nn.LazyLinear(len(space.choices))
    raise NotImplementedError()


def create_value_head(d_model: int) -> nn.Module:
    value_head = nn.Linear(d_model, 1)
    value_head.weight.data.fill_(0.0)
    value_head.bias.data.fill_(0.0)
    return value_head


def create_action_heads(action_space: Dict[str, ActionSpace]) -> nn.ModuleDict:
    action_heads = {}
    for name, space in action_space.items():
        action_heads[name] = create_head_for(space)
    return nn.ModuleDict(action_heads)
