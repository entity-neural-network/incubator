import os
from typing import List, Optional

import numpy as np
from entity_gym.env import (
    CategoricalAction,
    CategoricalActionMask,
    CategoricalActionSpace,
)
from griddly import gd

from enn_zoo.griddly_env import create_env

init_path = os.path.dirname(os.path.realpath(__file__))


def test_griddly_wrapper() -> None:
    env_class = create_env(
        global_observer_type=gd.ObserverType.BLOCK_2D,
        yaml_file=os.path.join(init_path, "env_descriptions/test/test.yaml"),
    )

    env = env_class()

    # Check the observation space is being created correctly from the test environment
    observation_space = env.obs_space()
    assert len(observation_space.entities) == 2
    assert observation_space.global_features == [
        "test_global_variable",
    ]

    assert observation_space.entities["entity_1"].features == [
        "x",
        "y",
        "z",
        "ox",
        "oy",
        "playerId",
        "entity_1_variable",
    ]
    assert observation_space.entities["entity_2"].features == ["x", "y", "z"]

    # Check the action space is being created correctly for the test environment
    action_space = env.action_space()
    assert isinstance(action_space["flat"], CategoricalActionSpace)
    assert action_space["flat"].index_to_label == [
        "NOP",
        "Left",
        "Up",
        "Right",
        "Down",
        "Do a little dance",
        "Make a little love",
        "Get down tonight",
    ]

    # Check that observation is created correctly
    observation = env.reset()

    # Check the entities in the observation
    assert np.all(
        observation.features["entity_1"]
        == np.array([[2, 2, 0, 0, 0, 1, 5]], dtype=np.float32)
    )

    print(np.sort(observation.features["entity_2"], axis=0))
    print(np.array([[2, 3, 0], [4, 4, 0]], dtype=np.float32))
    assert np.all(
        np.sort(observation.features["entity_2"], axis=0)
        == np.array([[2, 3, 0], [4, 4, 0]], dtype=np.float32)
    )

    # Check the masks in the observation
    assert isinstance(observation.actions["flat"], CategoricalActionMask)
    assert np.all(
        observation.actions["flat"].mask
        == np.array([[1, 1, 1, 1, 0, 1, 1, 1]])  # can do everything but move down
    )


def test_single_agent() -> None:
    """
    Create an environment and perform different action types to make sure the commands are translated
    correctly between griddly and enn wrappers
    """
    env_cls = create_env(
        global_observer_type=gd.ObserverType.BLOCK_2D,
        yaml_file=os.path.join(init_path, "env_descriptions/test/test_actions.yaml"),
    )
    env = env_cls()

    observation = env.reset()

    entity1_id = observation.ids["entity_1"][0]
    entity2_ids = observation.ids["entity_2"]

    # The starting location
    assert env.entity_locations[entity1_id] == [2, 1]

    move_down_action = CategoricalAction(
        indices=np.array([4], dtype=int),
        actors=[entity1_id],
        index_to_label=[],
    )
    observation_1 = env.act({"flat": move_down_action})

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
        indices=np.array([8], dtype=int), actors=[entity1_id], index_to_label=[]
    )
    observation_2 = env.act({"flat": remove_down_action})

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
        global_observer_type=gd.ObserverType.BLOCK_2D,
        yaml_file=os.path.join(
            init_path, "env_descriptions/test/test_multi_entities_actions.yaml"
        ),
    )
    env = env_cls()

    env.reset()

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
        indices=np.array([4], dtype=int),
        actors=[target_entity_1_id],
        index_to_label=env.action_space()["move_entity_one"].index_to_label,  # type: ignore
    )
    move_entity_two = CategoricalAction(
        indices=np.array([2], dtype=int),
        actors=[target_entity_2_id],
        index_to_label=env.action_space()["move_entity_two"].index_to_label,  # type: ignore
    )

    observation_1 = env.act(
        {"move_entity_one": move_entity_one, "move_entity_two": move_entity_two}
    )

    assert len(observation_1.ids["entity_1"]) == 3
    assert len(observation_1.ids["entity_2"]) == 3

    assert env.entity_locations[target_entity_1_id] == [1, 2]
    assert env.entity_locations[target_entity_2_id] == [3, 2]

    # Remove entity 1 and remove entity 2
    remove_entity_one = CategoricalAction(
        indices=np.array([2], dtype=int),
        actors=[target_entity_2_id],
        index_to_label=env.action_space()["remove_entity_one"].index_to_label,  # type: ignore
    )
    remove_entity_two = CategoricalAction(
        indices=np.array([4], dtype=int),
        actors=[target_entity_1_id],
        index_to_label=env.action_space()["remove_entity_two"].index_to_label,  # type: ignore
    )

    observation_2 = env.act(
        {"remove_entity_one": remove_entity_one, "remove_entity_two": remove_entity_two}
    )

    assert len(observation_2.ids["entity_1"]) == 2
    assert len(observation_2.ids["entity_2"]) == 2
