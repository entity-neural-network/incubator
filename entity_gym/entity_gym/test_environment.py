from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.environment import EnvList, ParallelEnvList
from entity_gym.environment import (
    SelectEntityAction,
)


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
