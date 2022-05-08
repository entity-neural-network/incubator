"""Agent"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from entity_gym.env.environment import Action, Observation


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Tuple[Dict[str, Action], float]:
        pass


__all__ = ["Agent"]
