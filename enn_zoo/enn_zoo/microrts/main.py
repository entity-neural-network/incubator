import argparse
from typing import Dict, Type
from enn_zoo.microrts import GymMicrorts

from entity_gym.environment import (
    CategoricalAction,
    SelectEntityActionMask,
    Environment,
    ObsSpace,
    CategoricalActionSpace,
    Action,
    Observation,
    SelectEntityAction,
    SelectEntityActionSpace,
)
import numpy as np


def print_obs(obs: Observation, total_reward: float, obs_filter: ObsSpace) -> None:
    print(f"Reward: {obs.reward}")
    print(f"Total reward: {total_reward}")
    entity_index = 0
    # for entity_type, features in obs.features.items():
    #     for entity in range(len(features)):
    #         print(
    #             f"{obs.ids[entity_type][entity_index]}: {entity_type}({', '.join(map(lambda nv: nv[0] + '=' + str(nv[1]), zip(obs_filter.entities[entity_type].features, features[entity])))})"
    #         )
    #         entity_index += 1


if __name__ == "__main__":
    env_cls = GymMicrorts

    print(env_cls)
    env = env_cls()
    obs_space = env_cls.obs_space()
    actions = env_cls.action_space()
    obs = env.reset_filter(obs_space)
    total_reward = obs.reward
    while not obs.done:
        print_obs(obs, total_reward, obs_space)
        action: Dict[str, Action] = {}
        for action_name, action_mask in obs.actions.items():
            action_def = actions[action_name]
            if action_mask.actor_ids is not None:
                actor_ids = action_mask.actor_ids
            elif action_mask.actor_types is not None:
                actor_ids = [
                    id for atype in action_mask.actor_types for id in obs.ids[atype]
                ]
            else:
                actor_ids = obs.index_to_id(obs_space)

            for actor_id in actor_ids:
                if isinstance(action_def, CategoricalActionSpace):
                    # Prompt user for action
                    print(f"Choose {action_name} for {actor_id}")
                    for i, label in enumerate(action_def.choices):
                        print(f"{i}) {label}")
                    choice_id = int(input())
                    if action_name not in action:
                            action[action_name] = CategoricalAction(
                            actions=np.zeros(
                                (0, len(action_def.choices)), dtype=np.int64
                            ),
                            actors=[],
                        )
                    a = action[action_name]
                    assert isinstance(a, CategoricalAction)
                    a.actions = np.array(list(a.actions) + [choice_id])
                    a.actors = list(a.actors) + [actor_id]
                elif isinstance(action_def, SelectEntityActionSpace):
                    assert isinstance(action_mask, SelectEntityActionMask)
                    # Prompt user for entity
                    print(f"{action_name}")
                    if action_mask.actee_ids is not None:
                        print(
                            f"Selectable entities: {', '.join([str(id) for id in action_mask.actee_ids])}"
                        )
                    else:
                        print("Selectable entities: all")
                    entity_id = int(input())
                    if action_name not in action:
                        action[action_name] = SelectEntityAction([], [])
                    a = action[action_name]
                    assert isinstance(a, SelectEntityAction)
                    a.actors = list(a.actors) + [actor_id]
                    a.actees = list(a.actees) + [entity_id]
                else:
                    raise ValueError(f"Unknown action type {action_def}")
        obs = env.act_filter(action, obs_space)
        total_reward += obs.reward

    print_obs(obs, total_reward, obs_space)
    print("Episode finished")
