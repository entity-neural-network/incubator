import os
import numpy as np
from enn_zoo.griddly import create_env

init_path = os.path.dirname(os.path.realpath(__file__))


def test_griddly_wrapper() -> None:
    env_class = create_env(os.path.join(init_path, "env_descriptions/test.yaml"))

    # Check the observation space is being created correctly from the test environment
    observation_space = env_class.obs_space()
    assert len(observation_space.entities) == 3
    assert set(observation_space.entities["__global__"].features) == {
        "_steps",
        "test_global_variable",
    }

    # TODO: currently we pass all possible variables to each feature, this should be fixed once features API is in a more consistent state
    assert set(observation_space.entities["entity_1"].features) == {
        "x",
        "y",
        "z",
        "orientation",
        "player_id",
        "entity_2_variable",
        "entity_1_variable",
    }
    assert set(observation_space.entities["entity_2"].features) == {
        "x",
        "y",
        "z",
        "orientation",
        "player_id",
        "entity_2_variable",
        "entity_1_variable",
    }

    # Check the action space is being created correctly fro the test environment
    action_space = env_class.action_space()
    assert set(action_space["move_one"].choices) == {
        "NOP",
        "Left",
        "Up",
        "Right",
        "Down",
    }
    assert set(action_space["move_two"].choices) == {
        "NOP",
        "Do a little dance",
        "Make a little love",
        "Get down tonight",
    }

    # Check that observation is created correctly
    env = env_class()
    observation = env._make_observation()

    # Check the entities in the observation
    assert np.all(observation.entities["entity_1"] == np.array([[2, 2, 0, 0, 1, 0, 5]]))
    assert np.all(
        np.sort(observation.entities["entity_2"], axis=0)
        == np.array([[2, 3, 0, 0, 0, 10, 0], [4, 4, 0, 0, 0, 10, 0]])
    )

    # Check the masks in the observation
    # TODO: actually make this test have non-0 masks...
    assert observation.action_masks["move_one"].actors[0] == 0
    assert np.all(
        observation.action_masks["move_one"].mask == np.array([[0, 0, 0, 0, 0]])
    )
    assert observation.action_masks["move_two"].actors[0] == 0
    assert np.all(observation.action_masks["move_two"].mask == np.array([[0, 0, 0, 0]]))
