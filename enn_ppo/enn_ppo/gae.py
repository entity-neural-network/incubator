# adapted from https://github.com/vwxyzjn/cleanrl
from typing import (
    Tuple,
)
from enn_ppo.agent import PPOAgent
import torch

from entity_gym.environment import *
from entity_gym.simple_trace import Tracer
from .config import *


def returns_and_advantages(
    agent: PPOAgent,
    next_obs: VecObs,
    next_done: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gae: float,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    tracer: Tracer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # bootstrap value if not done
    next_value = agent.get_auxiliary_head(
        next_obs.features, next_obs.visible, "value", tracer
    ).reshape(1, -1)
    num_steps = values.size(0)
    if gae:
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return
        advantages = returns - values
    # Need to detach here because bug in pytorch that otherwise causes spurious autograd errors and memory leaks when dedicated value function network is used.
    # possibly same cause as this: https://github.com/pytorch/pytorch/issues/71495
    return returns.detach(), advantages.detach()
