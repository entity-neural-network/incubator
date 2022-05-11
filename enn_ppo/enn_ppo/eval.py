from typing import Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from entity_gym.env import *
from entity_gym.env.add_metrics_wrapper import AddMetricsWrapper
from entity_gym.serialization import SampleRecordingVecEnv
from entity_gym.simple_trace import Tracer
from torch.utils.tensorboard import SummaryWriter

from enn_ppo.agent import PPOAgent
from enn_ppo.config import EnvConfig, EvalConfig, RolloutConfig
from enn_ppo.rollout import Rollout


def run_eval(
    cfg: EvalConfig,
    env_cfg: EnvConfig,
    rollout: RolloutConfig,
    create_env: Callable[[EnvConfig, int, int, int], VecEnv],
    create_opponent: Callable[
        [str, ObsSpace, Mapping[str, ActionSpace], torch.device], PPOAgent
    ],
    agent: PPOAgent,
    device: torch.device,
    tracer: Tracer,
    writer: Optional[SummaryWriter],
    global_step: int,
    rank: int,
    parallelism: int,
) -> None:
    # TODO: metrics are biased towards short episodes
    processes = cfg.processes or rollout.processes
    num_envs = cfg.num_envs or rollout.num_envs

    metric_filter: Optional[npt.NDArray[np.bool8]] = None

    envs: VecEnv = AddMetricsWrapper(
        create_env(
            cfg.env or env_cfg,
            num_envs // parallelism,
            processes,
            rank * num_envs // parallelism,
        ),
        metric_filter,
    )
    obs_space = envs.obs_space()
    action_space = envs.action_space()

    assert num_envs % parallelism == 0, (
        "Number of eval environments must be divisible by parallelism: "
        f"{num_envs} % {parallelism} = {num_envs % parallelism}"
    )

    if cfg.opponent is not None:
        opponent = create_opponent(cfg.opponent, obs_space, action_space, device)
        if cfg.opponent_only:
            agents: Union[
                PPOAgent, List[Tuple[npt.NDArray[np.int64], PPOAgent]]
            ] = opponent
        else:
            agents = [
                (np.array([2 * i for i in range(num_envs // parallelism // 2)]), agent),
                (
                    np.array([2 * i + 1 for i in range(num_envs // parallelism // 2)]),
                    opponent,
                ),
            ]
            metric_filter = np.arange(num_envs // parallelism) % 2 == 0
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

    if parallelism > 1:
        for metric in metrics.values():
            tcount = torch.tensor(metric.count)
            tsum = torch.tensor(metric.sum)
            tmax = torch.tensor(metric.max)
            tmin = torch.tensor(metric.min)
            dist.all_reduce(tcount, op=dist.ReduceOp.SUM)
            dist.all_reduce(tsum, op=dist.ReduceOp.SUM)
            dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
            dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
            metric.count = int(tcount.item())
            metric.sum = tsum.item()
            metric.max = tmax.item()
            metric.min = tmin.item()
    if writer is not None:
        if cfg.capture_videos:
            # save the videos
            writer.add_video(
                f"eval/video",
                torch.tensor(eval_rollout.rendered).permute(1, 0, 4, 2, 3),
                global_step,
                fps=30,
            )

        for name, value in metrics.items():
            writer.add_scalar(f"eval/{name}.mean", value.mean, global_step)
            writer.add_scalar(f"eval/{name}.min", value.min, global_step)
            writer.add_scalar(f"eval/{name}.max", value.max, global_step)
            writer.add_scalar(f"eval/{name}.count", value.count, global_step)
    print(
        f"[eval] global_step={global_step} {'  '.join(f'{name}={value.mean}' for name, value in metrics.items())}"
    )
    envs.close()
