from dataclasses import dataclass
from rogue_net.rogue_net import RogueNet, RogueNetConfig
import torch
import json
from typing import Mapping, Optional
from enn_ppo.agent import PPOAgent
from enn_ppo.train import train
from enn_zoo import griddly_env
import hyperstate
import enn_ppo.config as config
from entity_gym.environment import *
from entity_gym.examples import ENV_REGISTRY
from enn_zoo.griddly_env import GRIDDLY_ENVS
from enn_zoo.codecraft.cc_vec_env import codecraft_env_class, CodeCraftVecEnv
from enn_zoo.codecraft.codecraftnet.adapter import CCNetAdapter
from enn_zoo.microrts import GymMicrorts


@dataclass
class TrainConfig(config.TrainConfig):
    """Experiment settings.

    Attributes:
        codecraft_net: if toggled, use the DeepCodeCraft policy network instead of RogueNet (only works with CodeCraft environment)
    """

    codecraft_net: bool = False


def create_cc_env(cfg: config.EnvConfig, num_envs: int, num_processes: int) -> VecEnv:
    return CodeCraftVecEnv(num_envs, json.loads(cfg.kwargs))


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
        env_cls = codecraft_env_class(objective)
    elif cfg.env.id == "GymMicrorts":
        env_cls = GymMicrorts
    else:
        raise KeyError(
            f"Unknown gym_id: {cfg.env.id}\nAvailable environments: {list(ENV_REGISTRY.keys()) + list(GRIDDLY_ENVS.keys()) + ['CodeCraft']}"
        )

    agent: Optional[PPOAgent] = None
    if cfg.codecraft_net:
        agent = CCNetAdapter(device)  # type: ignore

    train(
        cfg=cfg,
        env_cls=env_cls,
        agent=agent,
        create_env=create_cc_env if cfg.env.id == "CodeCraft" else None,
    )


if __name__ == "__main__":
    main()
