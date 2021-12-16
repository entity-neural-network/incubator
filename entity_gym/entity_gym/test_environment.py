from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.environment import EnvList, ParallelEnvList
from entity_gym.environment import (
    SelectEntityAction,
    Observation,
    ObsSpace,
    Entity,
    batch_obs,
    merge_obs
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

    observation1 = Observation(
        {
            "entity1": np.array([[10, 10, 10], [10, 10, 10]], np.float32)
        },
        ["entity1_0", "entity1_1"],
        {},
        0.0,
        False,
        None
    )

    observation2 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0"],
        {},
        0.0,
        False,
        None
    )

    # now we introduce the rare entity, and remove entity1
    observation3 = Observation(
        {
            "rare": np.array([[10, 10, 10, 4, 2], [10, 10, 10, 4, 2], [10, 10, 10, 4, 2]], np.float32)
        },
        ["rare1_0", "rare1_1", "rare1_2"],
        {},
        0.0,
        False,
        None
    )

    obs_batch = batch_obs([observation1, observation2, observation3], obs_space)

    # entity1 observations should have a ragged array with lengths [2, 1, 0]
    # rare observations should have a ragged array with lengths [0, 0, 3]
    assert np.all(obs_batch.entities['entity1'].size1() == [2, 1, 0])
    assert np.all(obs_batch.entities['rare'].size1() == [0, 0, 3])


def test_batch_obs_select_entity_action():
    """
    We  have a set of observations and only a single one of those observations contains a paricular action and associated mask.
    When this action is batched, it needs to needs to contain 0-length rows for that action/mask for all other observations.
    """

    obs_space = ObsSpace(
        {
            "entity1": Entity(["x", "y", "z"]),
            "entity2": Entity(["x", "y", "z"]),
            "entity3": Entity(["x", "y", "z"]),
        }
    )

    observation1 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0", "entity2_0"],
        {
            # entity1 can low five entity 2 and vice versa
            "low-five": SelectEntityAction([("entity1_0", "entity2_0"), ("entity2_0", "entity1_0")])
        },
        0.0,
        False,
        None
    )

    observation2 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0","entity3_0"],
        {
            # entity3 can high five entity 1
            "high-five": SelectEntityAction([("entity3_0", "entity1_0")])
        },
        0.0,
        False,
        None
    )

    # now we introduce the rare entity, and remove entity1
    observation3 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32),
            "entity3": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0", "entity3_0"],
        {
            # entity3 can high five entity 1, and entity 2 can mid five entity3. entity1 and entity2 can low five each other
            "high-five": SelectEntityAction([("entity3_0", "entity1_0")]),
            "mid-five": SelectEntityAction([("entity2_0", "entity3_0")]),
            "low-five": SelectEntityAction([("entity1_0", "entity2_0"), ("entity2_0", "entity1_0")])
        },
        0.0,
        False,
        None
    )

    obs_batch = batch_obs([observation1, observation2, observation3], obs_space)

    # entity1 observations should have a ragged array with lengths [2, 1, 0]
    # rare observations should have a ragged array with lengths [0, 0, 3]
    assert np.all(obs_batch.entities['entity1'].size1() == [2, 1, 0])
    assert np.all(obs_batch.entities['rare'].size1() == [0, 0, 3])

def test_batch_obs_categorical_action():
    """
    We  have a set of observations and only a single one of those observations contains a paricular action and associated mask.
    When this action is batched, it needs to needs to contain 0-length rows for that action/mask for all other observations.
    """

    obs_space = ObsSpace(
        {
            "entity1": Entity(["x", "y", "z"]),
            "entity2": Entity(["x", "y", "z"]),
            "entity3": Entity(["x", "y", "z"]),
        }
    )

    observation1 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0", "entity2_0"],
        {
            # entity1 can high five entity 2 and vice versa
            "high-five": SelectEntityAction([("entity1_0", "entity2_0"), ("entity2_0", "entity1_0")])
        },
        0.0,
        False,
        None
    )

    observation2 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0","entity3_0"],
        {
            # entity3 can high five entity 1
            "high-five": SelectEntityAction([("entity3_0", "entity1_0")])
        },
        0.0,
        False,
        None
    )

    # now we introduce the rare entity, and remove entity1
    observation3 = Observation(
        {
            "entity1": np.array([[10, 10, 10]], np.float32),
            "entity2": np.array([[10, 10, 10]], np.float32),
            "entity3": np.array([[10, 10, 10]], np.float32)
        },
        ["entity1_0", "entity3_0"],
        {
            # entity3 can high five entity 1, and entity 2 can high five entity3. entity1 and entity2 cannot high five each other
            "high-five": SelectEntityAction([("entity3_0", "entity1_0"), ("entity2_0", "entity3_0")])
        },
        0.0,
        False,
        None
    )

    obs_batch = batch_obs([observation1, observation2, observation3], obs_space)

    # entity1 observations should have a ragged array with lengths [2, 1, 0]
    # rare observations should have a ragged array with lengths [0, 0, 3]
    assert np.all(obs_batch.entities['entity1'].size1() == [2, 1, 0])
    assert np.all(obs_batch.entities['rare'].size1() == [0, 0, 3])

# def test_merge_rare_entity():


# def test_merge_rare_action():
