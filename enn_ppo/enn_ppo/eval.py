from enn_ppo.agent import PPOAgent
from entity_gym.environment import *
from entity_gym.serialization import SampleRecordingVecEnv
from entity_gym.simple_trace import Tracer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List, Mapping, Tuple, Type, Union
import numpy as np
import numpy.typing as npt
import torch

from enn_ppo.config import EnvConfig, EvalConfig, RolloutConfig
from enn_ppo.rollout import Rollout


def run_eval(
    cfg: EvalConfig,
    env_cfg: EnvConfig,
    rollout: RolloutConfig,
    env_cls: Type[Environment],
    create_env: Callable[[EnvConfig, int, int], VecEnv],
    create_opponent: Callable[
        [str, ObsSpace, Mapping[str, ActionSpace], torch.device], PPOAgent
    ],
    agent: PPOAgent,
    device: torch.device,
    tracer: Tracer,
    writer: SummaryWriter,
    global_step: int,
) -> None:
    # TODO: metrics are biased towards short episodes
    processes = cfg.processes or rollout.processes
    num_envs = cfg.num_envs or rollout.num_envs
    envs = create_env(env_cfg, num_envs, processes)
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()

    if cfg.opponent is not None:
        opponent = create_opponent(cfg.opponent, obs_space, action_space, device)
        if cfg.opponent_only:
            agents: Union[
                PPOAgent, List[Tuple[npt.NDArray[np.int64], PPOAgent]]
            ] = opponent
        else:
            agents = [
                (np.array([2 * i for i in range(num_envs // 2)]), agent),
                (
                    np.array([2 * i + 1 for i in range(num_envs // 2)]),
                    opponent,
                ),
            ]
    else:
        agents = agent

    if cfg.capture_samples:
        envs = SampleRecordingVecEnv(
            envs, cfg.capture_samples, cfg.capture_samples_subsample
        )
    eval_rollout = Rollout(
        envs,
        obs_space=obs_space,
        action_space=action_space,
        agent=agents,
        device=device,
        tracer=tracer,
    )
    _, _, metrics = eval_rollout.run(
        cfg.steps,
        record_samples=False,
        capture_videos=cfg.capture_videos,
        capture_logits=cfg.capture_logits,
    )

    if cfg.capture_videos:
        # save the videos
        writer.add_video(
            f"eval/video",
            torch.tensor(eval_rollout.rendered).permute(1, 0, 4, 2, 3),
            global_step,
            fps=30,
        )

    for name, value in metrics.items():
        writer.add_scalar(f"eval/{name}", value, global_step)
    print(
        f"[eval] global_step={global_step} {'  '.join(f'{name}={value}' for name, value in metrics.items())}"
    )
    envs.close()
