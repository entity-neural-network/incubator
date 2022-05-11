import functools
from typing import Any, Dict, Type

import numpy as np
from entity_gym.env import Observation

from enn_zoo.griddly_env.wrappers.griddly_env import GriddlyEnv


def grafter_env(**kwargs: Any) -> Type[GriddlyEnv]:
    achievement_names = [
        "ach_collect_wood",
        "ach_collect_coal",
        "ach_collect_diamond",
        "ach_collect_drink",
        "ach_collect_iron",
        "ach_collect_sapling",
        "ach_collect_stone",
        "ach_collect_wood",
        "ach_defeat_skeleton",
        "ach_defeat_zombie",
        "ach_eat_cow",
        "ach_eat_plant",
        "ach_make_iron_pickaxe",
        "ach_make_iron_sword",
        "ach_make_stone_pickaxe",
        "ach_make_stone_sword",
        "ach_make_wood_pickaxe",
        "ach_make_wood_sword",
        "ach_place_furnace",
        "ach_place_plant",
        "ach_place_stone",
        "ach_place_table",
        "ach_wake_up",
    ]

    def __do_init__(self, **kwargs: Any) -> None:  # type: ignore
        GriddlyEnv.__init__(self, **kwargs)
        self.total_episodes = 0
        self.achievement_counter = {n: 0 for n in achievement_names}

    class InstantiatedGrafterEnv(GriddlyEnv):
        __init__ = functools.partialmethod(__do_init__, **kwargs)  # type: ignore

        def _add_grafter_metrics(self, observation: Observation) -> Observation:
            # If the env is done, calculate the grafter score
            if observation.done:
                achievements = self._env.game.get_global_variable(achievement_names)

                for ach_name, values in achievements.items():
                    self.achievement_counter[ach_name] += values[1]  # type: ignore

                self.total_episodes += 1  # type: ignore

                sum_log = 0
                for ach_name, counter in self.achievement_counter.items():  # type: ignore
                    sum_log += np.log(1 + 100 * counter / self.total_episodes)  # type: ignore

                crafter_score = np.exp(sum_log / len(achievement_names)) - 1

                observation.metrics = {"crafter_score": crafter_score}

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
