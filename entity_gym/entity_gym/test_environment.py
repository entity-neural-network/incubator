from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.environment import EnvList, ParallelEnvList
from entity_gym.environment import (
    SelectEntityActionSpace,
    CategoricalActionSpace,
    DenseSelectEntityActionMask,
    DenseCategoricalActionMask,
    SelectEntityAction,
    Observation,
    ObsSpace,
    Entity,
    batch_obs,
    merge_obs,
)
import numpy as np


def test_env_list() -> None:
    env_cls = CherryPick

    obs_space = env_cls.obs_space()

    # 100 environments
    envs = EnvList(env_cls, {}, 100)

    obs_reset = envs.reset(obs_space)
    assert len(obs_reset.ids) == 100

    actions = [
                  {"Pick Cherry": SelectEntityAction(actions=[("Player", "Cherry 1")])}
              ] * 100
    obs_act = envs.act(actions, obs_space)

    assert len(obs_act.ids) == 100


def test_parallel_env_list() -> None:
    env_cls = CherryPick

    obs_space = env_cls.obs_space()

    # 100 environments split across 10 processes
    envs = ParallelEnvList(env_cls, {}, 100, 10)

    obs_reset = envs.reset(obs_space)
    assert len(obs_reset.ids) == 100

    actions = [
                  {"Pick Cherry": SelectEntityAction(actions=[("Player", "Cherry 1")])}
              ] * 100
    obs_act = envs.act(actions, obs_space)

    assert len(obs_act.ids) == 100


def test_batch_obs_entities():
    """
    We  have a set of observations and only a single one of those observations contains a paricular entity.
    When this entity is batched, it needs to needs to contain 0-length rows for that entity for all other observations.
    """

    obs_space = ObsSpace(
        {
            "entity1": Entity(["x", "y", "z"]),
            "rare": Entity(["x", "y", "z", "health", "thing"]),
        }
    )

    action_space = {}

    observation1 = Observation(
        {"entity1": np.array([[10, 10, 10], [10, 10, 10]], np.float32)},
        ["entity1_0", "entity1_1"],
        {},
        0.0,
        False,
        None,
    )

    observation2 = Observation(
        {"entity1": np.array([[10, 10, 10]], np.float32)},
        ["entity1_0"],
        {},
        0.0,
        False,
        None,
    )

    observation3 = Observation(
        {
            "rare": np.array(
                [[10, 10, 10, 4, 2], [10, 10, 10, 4, 2], [10, 10, 10, 4, 2]], np.float32
            )
        },
        ["rare1_0", "rare1_1", "rare1_2"],
        {},
        0.0,
        False,
        None,
    )

    obs_batch = batch_obs(
        [observation1, observation2, observation3], obs_space, action_space
    )

    # entity1 observations should have a ragged array with lengths [2, 1, 0]
    # rare observations should have a ragged array with lengths [0, 0, 3]
    assert np.all(obs_batch.entities["entity1"].size1() == [2, 1, 0])
    assert np.all(obs_batch.entities["rare"].size1() == [0, 0, 3])


def test_batch_obs_select_entity_action():
    """
    We have three actions types that are dependent on the entities that are present in each observation
    This is common in procedurally generated environments, where types of objects/entities can be generated randomly,
    or in environments where there are possible rare interactions between entities.
    """

    obs_space = ObsSpace(
        {
            "entity1": Entity(["x", "y", "z"]),
            "entity2": Entity(["x", "y", "z"]),
            "entity3": Entity(["x", "y", "z"]),
        }
    )

    action_space = {
        "high_five": SelectEntityActionSpace(),
        "mid_five": SelectEntityActionSpace(),
        "low_five": SelectEntityActionSpace(),
    }

    observation1 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32),
        },
        ["entity1_0", "entity2_0"],
        {
            # entity1 can low five entity 2 and vice versa
            "low_five": DenseSelectEntityActionMask(
                np.array([0, 1]), np.array([1, 0]), None
            )
        },
        0.0,
        False,
        None,
    )

    observation2 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity3": np.array([[10, 10, 10]], np.float32),
        },
        ["entity1_0", "entity3_0"],
        {
            # entity3 can high five entity 1
            "high_five": DenseSelectEntityActionMask(np.array([1]), np.array([0]), None)
        },
        0.0,
        False,
        None,
    )

    observation3 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10], [10, 10, 10]], np.float32),
            "entity3": np.array([[10, 10, 10]], np.float32),
        },
        ["entity1_0", "entity2_0", "entity2_1", "entity3_0"],
        {
            # entity3 can high five entity 1, and entity 2_0 and entity 2_1 can mid five entity3. entity1 and entity2 can low five each other
            "high_five": DenseSelectEntityActionMask(
                np.array([3]), np.array([0]), None
            ),
            "mid_five": DenseSelectEntityActionMask(np.array([1, 2]), np.array([3]), None),
            "low_five": DenseSelectEntityActionMask(
                np.array([0, 1, 2]), np.array([0, 1, 2]), None
            ),
        },
        0.0,
        False,
        None,
    )

    obs_batch = batch_obs(
        [observation1, observation2, observation3], obs_space, action_space
    )

    assert np.all(obs_batch.action_masks["high_five"].actors.size1() == [0, 1, 1])
    assert np.all(obs_batch.action_masks["high_five"].actees.size1() == [0, 1, 1])

    assert np.all(obs_batch.action_masks["mid_five"].actors.size1() == [0, 0, 2])
    assert np.all(obs_batch.action_masks["mid_five"].actees.size1() == [0, 0, 1])

    assert np.all(obs_batch.action_masks["low_five"].actors.size1() == [2, 0, 3])
    assert np.all(obs_batch.action_masks["low_five"].actees.size1() == [2, 0, 3])


def test_batch_obs_categorical_action():
    """
    In some cases there are categorical that may only exist for certain entity types, or may only exist under certain circumstances.
    A particular example would be an action that is only available when an entity has a particular state (a special item or similar)
    """

    obs_space = ObsSpace(
        {
            "entity1": Entity(["x", "y", "z"]),
            "entity2": Entity(["x", "y", "z"]),
            "entity3": Entity(["x", "y", "z"]),
        }
    )

    action_space = {
        "move": CategoricalActionSpace(["up", "down", "left", "right"]),
        "choose_inventory_item": CategoricalActionSpace(
            ["axe", "sword", "pigeon"]
        ),
    }

    observation1 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32),
        },
        ["entity1_0", "entity2_0"],
        {
            # both entity1 and entity2 can move all directions
            "move": DenseCategoricalActionMask(np.array([0, 1]), np.array([[True, True, True, True], [True, True, True, True]])),
        },
        0.0,
        False,
        None,
    )

    observation2 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity3": np.array([[10, 10, 10], [10, 10, 10]], np.float32),
        },
        ["entity1_0", "entity3_0", "entity3_1"],
        {
            # all entities can move. Entity 3_1 can also choose items
            "move": DenseCategoricalActionMask(np.array([0, 1, 2]), np.array([[True, True, True, True], [True, True, True, True], [True, True, True, True]])),
            "choose_inventory_item": DenseCategoricalActionMask(np.array([2]), np.array([[True, True, True]]))
        },
        0.0,
        False,
        None,
    )

    observation3 = Observation(
        {
            "entity1": np.array([[10, 10, 10], [10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10], [10, 10, 10]], np.float32),
            "entity3": np.array([[10, 10, 10], [10, 10, 10]], np.float32),
        },
        ["entity1_0", "entity1_1", "entity2_0", "entity2_1", "entity3_0", "entity3_1"],
        {
            # no entities can move or do anything
        },
        0.0,
        False,
        None,
    )

    obs_batch = batch_obs(
        [observation1, observation2, observation3], obs_space, action_space
    )

    assert np.all(obs_batch.action_masks["move"].actors.size1() == [2, 3, 0])
    assert np.all(obs_batch.action_masks["move"].masks.size1() == [2, 3, 0])

    assert np.all(obs_batch.action_masks["choose_inventory_item"].actors.size1() == [0, 1, 0])
    assert np.all(obs_batch.action_masks["choose_inventory_item"].masks.size1() == [0, 1, 0])

# def test_merge_rare_entity():


# def test_merge_rare_action():
