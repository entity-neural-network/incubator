import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Mapping, Optional

import hyperstate
import torch
import web_pdb

import enn_ppo.config as config
from enn_ppo.agent import PPOAgent
from enn_ppo.train import train
from enn_zoo import griddly_env
from enn_zoo.codecraft.cc_vec_env import CodeCraftVecEnv, codecraft_env_class
from enn_zoo.codecraft.codecraftnet.adapter import CCNetAdapter
from enn_zoo.griddly_env import GRIDDLY_ENVS
from enn_zoo.microrts import GymMicrorts
from enn_zoo.procgen_env.big_fish import BigFish
from entity_gym.environment import *
from entity_gym.examples import ENV_REGISTRY
from rogue_net.rogue_net import RogueNet, RogueNetConfig


@dataclass
class TrainConfig(config.TrainConfig):
    """Experiment settings.

    Attributes:
        codecraft_net: if toggled, use the DeepCodeCraft policy network instead of RogueNet (only works with CodeCraft environment)
    """

    codecraft_net: bool = False
    webpdb: bool = False


def create_cc_env(cfg: config.EnvConfig, num_envs: int, num_processes: int) -> VecEnv:
    kwargs = json.loads(cfg.kwargs)
    return CodeCraftVecEnv(
        num_envs,
        kwargs.get("objective", "ALLIED_WEALTH"),
        hidden_obs=kwargs.get("hidden_obs", False),
    )


def load_codecraft_policy(
    path: str,
    obs_space: ObsSpace,
    action_space: Mapping[str, ActionSpace],
    device: torch.device,
) -> PPOAgent:
    if path == "random":
        return RogueNet(
            RogueNetConfig(),
            obs_space,
            dict(action_space),
            regression_heads={"value": 1},
        ).to(device)
    else:
        return CCNetAdapter(str(device), load_from=path)


@hyperstate.command(TrainConfig)
def main(cfg: TrainConfig) -> None:
    if cfg.env.id in ENV_REGISTRY:
        env_cls = ENV_REGISTRY[cfg.env.id]
    elif cfg.env.id in GRIDDLY_ENVS:
        env_cls = griddly_env.create_env(**GRIDDLY_ENVS[cfg.env.id])
    elif cfg.env.id == "CodeCraft":
        objective = json.loads(cfg.env.kwargs).get("objective", "ALLIED_WEALTH")
        hidden_obs = json.loads(cfg.env.kwargs).get("hidden_obs", False)
        env_cls = codecraft_env_class(objective, hidden_obs)
    elif cfg.env.id == "GymMicrorts":
        env_cls = GymMicrorts
    elif cfg.env.id == "BigFish":
        env_cls = BigFish
    else:
        try:
            from enn_zoo import vizdoom_env
            from enn_zoo.vizdoom_env import VIZDOOM_ENVS

            env_cls = vizdoom_env.create_vizdoom_env(VIZDOOM_ENVS[cfg.env.id])
        except ImportError:
            raise KeyError(
                f"Unknown gym_id: {cfg.env.id}\nAvailable environments: {list(ENV_REGISTRY.keys()) + list(GRIDDLY_ENVS.keys()) + ['CodeCraft']}"
            )

    agent: Optional[PPOAgent] = None
    if cfg.codecraft_net:
        agent = CCNetAdapter(device)  # type: ignore

    with ExitStack() as stack:
        if cfg.webpdb:
            stack.enter_context(web_pdb.catch_post_mortem())
        train(
            cfg=cfg,
            env_cls=env_cls,
            agent=agent,
            create_env=create_cc_env if cfg.env.id == "CodeCraft" else None,
        )


if __name__ == "__main__":
    main()
