# adapted from https://github.com/vwxyzjn/cleanrl
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type

import hyperstate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from enn_ppo.agent import PPOAgent
from enn_ppo.config import *
from enn_ppo.eval import run_eval
from enn_ppo.gae import returns_and_advantages
from enn_ppo.ppo import ppo_loss, value_loss
from enn_ppo.rollout import Rollout
from entity_gym.environment import *
from entity_gym.environment.add_metrics_wrapper import AddMetricsWrapper
from entity_gym.examples import ENV_REGISTRY
from entity_gym.serialization import SampleRecordingVecEnv
from entity_gym.simple_trace import Tracer
from rogue_net.rogue_net import RogueNet


def _env_factory(env_cls: Type[Environment]) -> Callable[[EnvConfig, int, int], VecEnv]:
    def _create_env(cfg: EnvConfig, num_envs: int, processes: int) -> VecEnv:
        kwargs = json.loads(cfg.kwargs)
        if processes > 1:
            return ParallelEnvList(env_cls, kwargs, num_envs, processes)
        else:
            return EnvList(env_cls, kwargs, num_envs)

    return _create_env


def create_random_opponent(
    path: str,
    obs_space: ObsSpace,
    action_space: Mapping[str, ActionSpace],
    device: torch.device,
) -> PPOAgent:
    return RogueNet(
        RogueNetConfig(),
        obs_space,
        dict(action_space),
        regression_heads={"value": 1},
    ).to(device)


def train(
    cfg: TrainConfig,
    env_cls: Type[Environment],
    agent: Optional[PPOAgent] = None,
    create_env: Optional[Callable[[EnvConfig, int, int], VecEnv]] = None,
    create_opponent: Optional[
        Callable[[str, ObsSpace, Mapping[str, ActionSpace], torch.device], PPOAgent]
    ] = None,
) -> float:
    run_name = f"{cfg.env.id}__{cfg.name}__{cfg.seed}__{int(time.time())}"

    config = asdict(cfg)
    if os.path.exists("/xprun/info/config.ron"):
        import xprun  # type: ignore

        xp_info = xprun.current_xp()
        config["name"] = xp_info.xp_def.name
        config["base_name"] = xp_info.xp_def.base_name
        config["id"] = xp_info.id
        if "-" in xp_info.xp_def.name and xp_info.xp_def.name.split("-")[-1].isdigit():
            cfg.seed = int(xp_info.xp_def.name.split("-")[-1])
            config["seed"] = cfg.seed
        run_name = xp_info.xp_def.name
        out_dir: Optional[str] = os.path.join(
            "/mnt/xprun",
            xp_info.xp_def.project,
            xp_info.sanitized_name + "-" + xp_info.id,
        )
        Path(str(out_dir)).mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    data_path = Path(cfg.data_dir).absolute()
    data_path.mkdir(parents=True, exist_ok=True)
    data_dir = str(data_path)

    if cfg.track:
        import wandb

        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            dir=data_dir,
        )
    writer = SummaryWriter(os.path.join(data_dir, f"runs/{run_name}"))

    def flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flattened = {}
        for k, v in config.items():
            if isinstance(v, dict):
                flattened.update(flatten(v, k if prefix == "" else f"{prefix}.{k}"))
            else:
                flattened[prefix + k] = v
        return flattened

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in flatten(config).items()])),
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    cuda = torch.cuda.is_available() and cfg.cuda
    device = torch.device("cuda" if cuda else "cpu")
    tracer = Tracer(cuda=cuda)

    if create_env is None:
        create_env = _env_factory(env_cls)
    envs: VecEnv = AddMetricsWrapper(
        create_env(cfg.env, cfg.rollout.num_envs, cfg.rollout.processes)
    )
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()

    if cfg.capture_samples:
        if out_dir is None:
            sample_file = cfg.capture_samples
        else:
            sample_file = os.path.join(out_dir, cfg.capture_samples)
        envs = SampleRecordingVecEnv(envs, sample_file, cfg.capture_samples_subsample)

    if agent is None:
        agent = RogueNet(
            cfg.net,
            obs_space,
            action_space,
            regression_heads={"value": 1},
        ).to(device)
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=1e-5,
    )
    if cfg.track:
        wandb.watch(agent)

    if cfg.vf_net is not None:
        value_function: Optional[RogueNet] = RogueNet(
            cfg.vf_net,
            obs_space,
            action_space,
            regression_heads={"value": 1},
        ).to(device)
        vf_optimizer: Optional[optim.AdamW] = optim.AdamW(
            value_function.parameters(),  # type: ignore
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            eps=1e-5,
        )
    else:
        value_function = None
        vf_optimizer = None

    num_updates = cfg.total_timesteps // (cfg.rollout.num_envs * cfg.rollout.steps)

    rollout = Rollout(
        envs,
        obs_space=obs_space,
        action_space=action_space,
        agent=agent,
        value_function=value_function,
        device=device,
        tracer=tracer,
    )

    if cfg.eval is not None:
        if cfg.eval.run_on_first_step:
            next_eval_step: Optional[int] = 0
        else:
            next_eval_step = cfg.eval.interval
    else:
        next_eval_step = None

    def _run_eval() -> None:
        if cfg.eval is not None:
            assert create_env is not None
            assert agent is not None
            run_eval(
                cfg.eval,
                cfg.env,
                cfg.rollout,
                env_cls,
                create_env,
                create_opponent or create_random_opponent,
                agent,
                device,
                tracer,
                writer,
                rollout.global_step,
            )

    start_time = time.time()
    for update in range(1, num_updates + 1):
        if (
            cfg.eval is not None
            and next_eval_step is not None
            and rollout.global_step >= next_eval_step
        ):
            next_eval_step += cfg.eval.interval
            _run_eval()

        tracer.start("update")
        if (
            cfg.max_train_time is not None
            and time.time() - start_time >= cfg.max_train_time
        ):
            print("Max train time reached, stopping training.")
            break

        # Annealing the rate if instructed to do so.
        if cfg.optim.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            if cfg.max_train_time is not None:
                frac = min(
                    frac, max(0, 1.0 - (time.time() - start_time) / cfg.max_train_time)
                )
            lrnow = frac * cfg.optim.lr
            optimizer.param_groups[0]["lr"] = lrnow
            if vf_optimizer is not None:
                vf_optimizer.param_groups[0]["lr"] = lrnow

        tracer.start("rollout")

        next_obs, next_done, metrics = rollout.run(
            cfg.rollout.steps, record_samples=True, capture_logits=cfg.capture_logits
        )
        for name, value in metrics.items():
            writer.add_scalar(f"{name}.mean", value.mean, rollout.global_step)
            writer.add_scalar(f"{name}.max", value.max, rollout.global_step)
            writer.add_scalar(f"{name}.min", value.min, rollout.global_step)
            writer.add_scalar(f"{name}.count", value.count, rollout.global_step)

        # Double log these to remain compatible with old naming scheme
        # TODO: remove before release
        writer.add_scalar(
            "charts/episodic_return",
            metrics["episodic_reward"].mean,
            rollout.global_step,
        )
        writer.add_scalar(
            "charts/episodic_length",
            metrics["episode_length"].mean,
            rollout.global_step,
        )
        writer.add_scalar(
            "charts/episodes", metrics["episodic_reward"].count, rollout.global_step
        )
        writer.add_scalar("meanrew", metrics["reward"].mean, rollout.global_step)

        print(
            f"global_step={rollout.global_step} {'  '.join(f'{name}={value.mean}' for name, value in metrics.items())}"
        )

        values = rollout.values
        actions = rollout.actions
        entities = rollout.entities
        visible = rollout.visible
        action_masks = rollout.action_masks
        logprobs = rollout.logprobs
        global_step = rollout.global_step

        with torch.no_grad(), tracer.span("advantages"):
            returns, advantages = returns_and_advantages(
                value_function or agent,
                next_obs,
                next_done,
                rollout.rewards,
                rollout.dones,
                values,
                cfg.ppo.gae,
                cfg.ppo.gamma,
                cfg.ppo.gae_lambda,
                device,
                tracer,
            )

        # flatten the batch
        with tracer.span("flatten"):
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1).detach()
            b_values = values.reshape(-1).detach()

        tracer.end("rollout")

        # Optimize the policy and value network
        tracer.start("optimize")
        frames = cfg.rollout.num_envs * cfg.rollout.steps
        b_inds = np.arange(frames)
        clipfracs = []

        for epoch in range(cfg.optim.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, frames, cfg.optim.bs):
                end = start + cfg.optim.bs
                microbatch_size = (
                    cfg.optim.micro_bs
                    if cfg.optim.micro_bs is not None
                    else cfg.optim.bs
                )

                optimizer.zero_grad()
                if vf_optimizer is not None:
                    vf_optimizer.zero_grad()
                for _start in range(start, end, microbatch_size):
                    _end = _start + microbatch_size
                    mb_inds = b_inds[_start:_end]

                    b_entities = entities[mb_inds]
                    b_visible = visible[mb_inds]
                    b_action_masks = action_masks[mb_inds]
                    b_logprobs = logprobs[mb_inds]
                    b_actions = actions[mb_inds]
                    mb_advantages = b_advantages[mb_inds]  # type: ignore

                    with tracer.span("forward"):
                        (
                            _,
                            newlogprob,
                            entropy,
                            _,
                            aux,
                            _,
                        ) = agent.get_action_and_auxiliary(
                            b_entities,
                            b_visible,
                            b_action_masks,
                            prev_actions=b_actions,
                            tracer=tracer,
                        )
                        if value_function is None:
                            newvalue = aux["value"]
                        else:
                            newvalue = value_function.get_auxiliary_head(
                                b_entities, b_visible, "value", tracer=tracer
                            )

                    pg_loss, clipfrac, approx_kl = ppo_loss(
                        cfg.ppo, newlogprob, b_logprobs, mb_advantages, device, tracer
                    )
                    clipfracs += [clipfrac]

                    v_loss = value_loss(
                        cfg.ppo,
                        newvalue,
                        b_returns[mb_inds],  # type: ignore
                        b_values[mb_inds],  # type: ignore
                        tracer,
                    )

                    # TODO: what's correct way of combining entropy loss from multiple actions/actors on the same timestep?
                    if cfg.ppo.anneal_entropy:
                        frac = 1.0 - (update - 1.0) / num_updates
                        if cfg.max_train_time is not None:
                            frac = min(
                                frac,
                                max(
                                    0,
                                    1.0
                                    - (time.time() - start_time) / cfg.max_train_time,
                                ),
                            )
                        ent_coef = frac * cfg.ppo.ent_coef
                    else:
                        ent_coef = cfg.ppo.ent_coef
                    entropy_loss = torch.cat([e for e in entropy.values()]).mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef
                    loss *= microbatch_size / cfg.optim.bs

                    with tracer.span("backward"):
                        loss.backward()
                gradnorm = nn.utils.clip_grad_norm_(
                    agent.parameters(), cfg.optim.max_grad_norm
                )
                optimizer.step()
                if value_function is not None:
                    vf_gradnorm = nn.utils.clip_grad_norm_(
                        value_function.parameters(), cfg.optim.max_grad_norm
                    ).item()
                else:
                    vf_gradnorm = 0.0
                if vf_optimizer is not None:
                    vf_optimizer.step()

            if cfg.ppo.target_kl is not None:
                if approx_kl > cfg.ppo.target_kl:
                    break

        if cfg.cuda_empty_cache:
            torch.cuda.empty_cache()
        tracer.end("optimize")

        tracer.start("metrics")
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("charts/entropy_coef", ent_coef, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/gradnorm", gradnorm, global_step)
        writer.add_scalar("losses/vf_gradnorm", vf_gradnorm, global_step)
        for action_name, space in action_space.items():
            if isinstance(space, CategoricalActionSpace):
                _actions = actions.buffers[action_name].as_array().flatten()
                if len(_actions) > 0:
                    for i, label in enumerate(space.choices):
                        writer.add_scalar(
                            f"actions/{action_name}/{label}",
                            np.sum(_actions == i).item() / len(_actions),
                            global_step,
                        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        tracer.end("metrics")
        tracer.end("update")
        traces = tracer.finish()
        for callstack, timing in traces.items():
            writer.add_scalar(f"trace/{callstack}", timing, global_step)

    if cfg.eval is not None:
        _run_eval()

    envs.close()
    writer.close()

    return rollout.rewards.mean().item()


def _train(cfg: TrainConfig) -> float:
    return train(cfg, ENV_REGISTRY[cfg.env.id])


@hyperstate.command(TrainConfig)
def _main(cfg: TrainConfig) -> None:
    _train(cfg)


if __name__ == "__main__":
    _main()
