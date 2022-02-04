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
class NotHotdog(Environment):
    """
    On each timestep, there is either a generic "Object" entity with a `is_hotdog` property, or a "Hotdog" object.
    The "Player" entity is always present, and has an action to classify the other entity as hotdog or not hotdog.
    """

    @classmethod
    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            {
                "Player": Entity(["step"]),
                "Object": Entity(["is_hotdog"]),
                "Hotdog": Entity([]),
            }
        )

    @classmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        return {
            "classify": CategoricalActionSpace(["hotdog", "not_hotdog"]),
            "unused_action": CategoricalActionSpace(["0", "1"]),
        }

    def reset(self) -> Observation:
        self.step = 0
        self.is_hotdog = random.randint(0, 1)
        self.hotdog_object = random.randint(0, 1) == 1
        return self.observe()

    def act(self, action: Mapping[str, Action]) -> Observation:
        self.step += 1

        a = action["classify"]
        assert isinstance(a, CategoricalAction), f"{a} is not a CategoricalAction"
        if a.actions[0] == self.is_hotdog:
            reward = 1
        else:
            reward = 0
        done = True
        return self.observe(done, reward)

    def observe(self, done: bool = False, reward: float = 0) -> Observation:
        return Observation(
            features={
                "Player": np.array([[self.step,]], dtype=np.float32,),
                "Object": np.array([[self.is_hotdog,]], dtype=np.float32,)
                if (self.hotdog_object and self.is_hotdog == 0)
                or not self.hotdog_object
                else np.zeros((0, 1), dtype=np.float32).reshape(0, 1),
                "Hotdog": np.zeros((1, 0), dtype=np.float32)
                if self.hotdog_object and self.is_hotdog == 1
                else np.zeros((0, 0), dtype=np.float32),
            },
            actions={
                "classify": CategoricalActionMask(actor_ids=[0]),
                "unused_action": CategoricalActionMask(actor_ids=[]),
            },
            ids={"Player": [0]},
            reward=reward,
            done=done,
            end_of_episode_info=EpisodeStats(self.step, reward) if done else None,
        )
