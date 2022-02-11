import os
from typing import List, Optional

import numpy as np
from enn_zoo.griddly import create_env, GRIDDLY_ENVS
from entity_gym.environment import (
    CategoricalActionSpace,
    CategoricalActionMask,
    CategoricalAction,
)

init_path = os.path.dirname(os.path.realpath(__file__))


def test_griddly_wrapper() -> None:
    env_class = create_env(
        yaml_file=os.path.join(init_path, "env_descriptions/test/test.yaml")
    )

    # Check the observation space is being created correctly from the test environment
    observation_space = env_class.obs_space()
    assert len(observation_space.entities) == 3
    assert observation_space.entities["__global__"].features == [
        "_steps",
        "test_global_variable",
    ]

    # TODO: currently we pass all possible variables to each feature, this should be fixed once features API is in a more consistent state
    assert observation_space.entities["entity_1"].features == [
        "x",
        "y",
        "z",
        "ox",
        "oy",
        "player_id",
        "entity_1_variable",
    ]
    assert observation_space.entities["entity_2"].features == [
        "x",
        "y",
        "z",
        "ox",
        "oy",
        "player_id",
        "entity_2_variable",
    ]

    # Check the action space is being created correctly fro the test environment
    action_space = env_class.action_space()
    assert isinstance(action_space["move_one"], CategoricalActionSpace)
    assert action_space["move_one"].choices == [
        "NOP",
        "Left",
        "Up",
        "Right",
        "Down",
    ]
    assert isinstance(action_space["move_two"], CategoricalActionSpace)
    assert action_space["move_two"].choices == [
        "NOP",
        "Do a little dance",
        "Make a little love",
        "Get down tonight",
    ]

    # Check that observation is created correctly
    env = env_class()
    observation = env._make_observation()

    # Check the entities in the observation
    assert np.all(
        observation.features["entity_1"]
        == np.array([[2, 2, 0, 0, 0, 1, 5]], dtype=np.float32)
    )

    print(np.sort(observation.features["entity_2"], axis=0))
    print(np.array([[2, 3, 0, 0, 0, 0, 10], [4, 4, 0, 0, 0, 0, 10]], dtype=np.float32))
    assert np.all(
        np.sort(observation.features["entity_2"], axis=0)
        == np.array([[2, 3, 0, 0, 0, 0, 10], [4, 4, 0, 0, 0, 0, 10]], dtype=np.float32)
    )

    # Check the masks in the observation
    assert isinstance(observation.actions["move_one"], CategoricalActionMask)
    assert np.all(
        observation.actions["move_one"].mask
        == np.array([[1, 1, 1, 1, 0]])  # can do everything but move down
    )
    assert isinstance(observation.actions["move_two"], CategoricalActionMask)
    assert np.all(observation.actions["move_two"].mask == np.array([[1, 1, 1, 1]]))


def test_single_agent() -> None:
    """
    Create an environment and perform different action types to make sure the commands are translated
    correctly between griddly and enn wrappers
    """
    env_cls = create_env(
        yaml_file=os.path.join(init_path, "env_descriptions/test/test_actions.yaml")
    )
    env = env_cls()

    observation = env.reset()

    entity1_id = observation.ids["entity_1"][0]
    entity2_ids = observation.ids["entity_2"]

    # The starting location
    assert env.entity_locations[entity1_id] == [2, 1]

    move_down_action = CategoricalAction(
        actions=np.array([[4]], dtype=int), actors=[entity1_id]
    )
    observation_1 = env.act({"move_entity_one": move_down_action})

    # The entity has moved down
    assert len(observation_1.ids["entity_1"]) == 1
    assert env.entity_locations[entity1_id] == [2, 2]

    # There are three entity2 and one of them is in position 3,3
    assert len(observation_1.ids["entity_2"]) == 3
    assert (
        env.entity_locations[entity2_ids[0]] == [2, 3]
        or env.entity_locations[entity2_ids[1]] == [2, 3]
        or env.entity_locations[entity2_ids[2]] == [2, 3]
    )

    remove_down_action = CategoricalAction(
        actions=np.array([[4]], dtype=int), actors=[entity1_id]
    )
    observation_2 = env.act({"remove_entity_two": remove_down_action})

    assert len(observation_2.ids["entity_1"]) == 1

    # There are two entity_2 and none of them are in 3,3
    assert len(observation_2.ids["entity_2"]) == 2
    assert np.all(
        [env.entity_locations[id] != [2, 3] for id in observation_2.ids["entity_2"]]
    )


def test_single_agent_multi_entity() -> None:
    """
    Create an environment with multiple entities and perform different action types to make sure the commands are translated
    correctly between griddly and enn wrappers
    """

    env_cls = create_env(
        yaml_file=os.path.join(
            init_path, "env_descriptions/test/test_multi_entities_actions.yaml"
        )
    )
    env = env_cls()

    observation = env.reset()

    def get_id_by_location(location: List[int]) -> Optional[int]:
        for k, v in env.entity_locations.items():
            if v == location:
                assert isinstance(k, int)
                return k
        return None

    target_entity_1_id = get_id_by_location([1, 1])
    target_entity_2_id = get_id_by_location([3, 3])

    # Move target entity1 down and target entity 2 down
    move_entity_one = CategoricalAction(
        actions=np.array([[4]], dtype=int), actors=[target_entity_1_id]
    )
    move_entity_two = CategoricalAction(
        actions=np.array([[2]], dtype=int), actors=[target_entity_2_id]
    )

    observation_1 = env.act(
        {"move_entity_one": move_entity_one, "move_entity_two": move_entity_two}
    )

    assert len(observation_1.ids["entity_1"]) == 3
    assert len(observation_1.ids["entity_2"]) == 3

    assert env.entity_locations[target_entity_1_id] == [1, 2]
    assert env.entity_locations[target_entity_2_id] == [3, 2]

    # Remove entity 1 and remove entity 2
    remove_entity_two = CategoricalAction(
        actions=np.array([[4]], dtype=int), actors=[target_entity_1_id]
    )
    remove_entity_one = CategoricalAction(
        actions=np.array([[2]], dtype=int), actors=[target_entity_2_id]
    )

    observation_2 = env.act(
        {"remove_entity_one": remove_entity_one, "remove_entity_two": remove_entity_two}
    )

    assert len(observation_2.ids["entity_1"]) == 2
    assert len(observation_2.ids["entity_2"]) == 2
