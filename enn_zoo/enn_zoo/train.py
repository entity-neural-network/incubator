import json
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Type, Union

import enn_trainer.config as config
import hyperstate
import torch
import web_pdb
from enn_trainer.agent import PPOAgent
from enn_trainer.train import State, initialize, train
from entity_gym.env import *
from entity_gym.examples import ENV_REGISTRY
from hyperstate import StateManager

from enn_zoo import griddly_env
from enn_zoo.codecraft.cc_vec_env import CodeCraftVecEnv
from enn_zoo.codecraft.codecraftnet.adapter import CCNetAdapter
from enn_zoo.griddly_env import GRIDDLY_ENVS
from enn_zoo.microrts import GymMicrorts
from enn_zoo.procgen_env.big_fish import BigFish
from enn_zoo.procgen_env.boss_fight import BossFight
from enn_zoo.procgen_env.leaper import Leaper
from enn_zoo.procgen_env.plunder import Plunder
from enn_zoo.procgen_env.star_pilot import StarPilot
from rogue_net.rogue_net import RogueNet, RogueNetConfig


@dataclass
class TrainConfig(config.TrainConfig):
    """Experiment settings.

    Attributes:
        codecraft_net: if toggled, use the DeepCodeCraft policy network instead of RogueNet (only works with CodeCraft environment)
    """

    codecraft_net: bool = False
    webpdb: bool = False


def create_cc_env(
    cfg: config.EnvConfig, num_envs: int, num_processes: int, first_env_index: int
) -> VecEnv:
    kwargs = json.loads(cfg.kwargs)
    return CodeCraftVecEnv(
        num_envs,
        **kwargs,
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


@hyperstate.stateful_command(TrainConfig, State, initialize)
def main(state_manager: StateManager) -> None:
    cfg = state_manager.config
    if cfg.env.id in ENV_REGISTRY:
        env: Union[
            Type[Environment], Callable[[config.EnvConfig, int, int, int], VecEnv]
        ] = ENV_REGISTRY[cfg.env.id]
    elif cfg.env.id in GRIDDLY_ENVS:
        env = griddly_env.create_env(**GRIDDLY_ENVS[cfg.env.id])
    elif cfg.env.id == "CodeCraft":
        env = create_cc_env
    elif cfg.env.id == "GymMicrorts":
        env = GymMicrorts
    elif cfg.env.id.startswith("Procgen"):
        if cfg.env.id == "Procgen:BigFish":
            env = BigFish
        elif cfg.env.id == "Procgen:BossFight":
            env = BossFight
        elif cfg.env.id == "Procgen:StarPilot":
            env = StarPilot
        elif cfg.env.id == "Procgen:Leaper":
            env = Leaper
        elif cfg.env.id == "Procgen:Plunder":
            env = Plunder
        else:
            raise NotImplementedError(f"Unknown procgen env: {cfg.env.id}")
    else:
        try:
            from enn_zoo import vizdoom_env
            from enn_zoo.vizdoom_env import VIZDOOM_ENVS

            env = vizdoom_env.create_vizdoom_env(VIZDOOM_ENVS[cfg.env.id])
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
            state_manager=state_manager,
            env=env,
            agent=agent,
            create_opponent=load_codecraft_policy
            if cfg.env.id == "CodeCraft"
            else None,
        )


if __name__ == "__main__":
    main()
