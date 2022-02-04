from dataclasses import dataclass
import numpy as np
import random
from typing import Dict, Mapping

from entity_gym.environment import (
    CategoricalAction,
    CategoricalActionMask,
    Entity,
    Environment,
    CategoricalActionSpace,
    ActionSpace,
    EpisodeStats,
    ObsSpace,
    Observation,
    Action,
)


@dataclass
class MultiArmedBandit(Environment):
    """
    Task with single cateorical action with 5 choices which gives a reward of 1 for choosing action 0 and reward of 0 otherwise.
    """

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace({"MultiArmedBandit": Entity(["step"]),})

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "pull": CategoricalActionSpace(["A", "B", "C", "D", "E"]),
        }

    def reset(self) -> Observation:
        self.step = 0
        self._total_reward = 0
        return self.observe()

    def act(self, action: Mapping[str, Action]) -> Observation:
        self.step += 1

        a = action["pull"]
        assert isinstance(a, CategoricalAction), f"{a} is not a CategoricalAction"
        if a.actions[0] == 0:
            reward = 1
        else:
            reward = 0
        done = self.step >= 32
        self._total_reward += reward
        return self.observe(done, reward)

    def observe(self, done: bool = False, reward: float = 0) -> Observation:
        return Observation(
            features={"MultiArmedBandit": np.array([[self.step,]], dtype=np.float32,),},
            actions={"pull": CategoricalActionMask(actor_ids=[0]),},
            ids={"MultiArmedBandit": [0]},
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, self._total_reward)
            if done
            else None,
        )
