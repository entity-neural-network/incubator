import functools
from typing import Any, Dict, Type

from enn_zoo.griddly_env.wrappers.griddly_env import GriddlyEnv
from entity_gym.environment import Observation


def grafter_env(**kwargs: Any) -> Type[GriddlyEnv]:
    class InstantiatedGrafterEnv(GriddlyEnv):
        __init__ = functools.partialmethod(GriddlyEnv.__init__, **kwargs)  # type: ignore

        def _add_grafter_metrics(self, observation: Observation) -> Observation:
            # metrics
            return observation

        def make_observation(
            self,
            obs: Dict[str, Any],
            reward: int = 0,
            done: bool = False,
        ) -> Observation:
            observation = super().make_observation(obs, reward, done)
            return self._add_grafter_metrics(observation)

    return InstantiatedGrafterEnv  # type: ignore
