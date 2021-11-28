from typing import Callable, List, Literal
from torch import nn
from torch.nn.modules.linear import Linear


def relu_function_creator(num_layer: int) -> nn.Module:
    return nn.ReLU()


def mlp_backbone(
    d_model: int,
    nlayers: int,
    activation_function_creator: Callable[[int], nn.Module] = relu_function_creator,
) -> nn.Module:
    layers: List[nn.Module] = []
    for num_layer in range(nlayers):
        layers.append(Linear(d_model, d_model))
        layers.append(activation_function_creator(num_layer))
    return nn.Sequential(*layers)
