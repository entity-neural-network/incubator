# adapted from https://github.com/vwxyzjn/cleanrl
import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import time
from distutils.util import strtobool
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import json
from entity_gym.environment import *

import numpy as np
import numpy.typing as npt
from entity_gym.examples import ENV_REGISTRY
from enn_zoo.griddly import GRIDDLY_ENVS, create_env
from enn_zoo.codecraft.cc_vec_env import CodeCraftEnv, CodeCraftVecEnv
from entity_gym.serialization import SampleRecordingVecEnv
from enn_ppo.simple_trace import Tracer
from rogue_net.relpos_encoding import RelposEncodingConfig
from rogue_net.actor import AutoActor
from rogue_net import head_creator
from rogue_net.translate_positions import TranslatePositions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBuffer


def parse_args(override_args: Optional[List[str]] = None) -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MoveToOrigin",
        help='the id of the gym environment')
    parser.add_argument('--env-kwargs', type=str, default="{}",
        help='JSON dictionary with keyword arguments for the environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0,
        help='the weight decay of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
        help='total timesteps of the experiments')
    parser.add_argument('--max-train-time', type=int, default=None,
        help='train for at most this many seconds')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="enn-ppo",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default="entity-neural-network",
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-samples', type=str, default=None,
        help='if set, write the samples to this file')
    parser.add_argument('--capture-logits', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='If --capture-samples is set, record full logits of the agent')
    parser.add_argument('--processes', type=int, default=1,
        help='The number of processes to use to collect env data. The envs are split as equally as possible across the processes')
    parser.add_argument('--trial', type=int, default=None,
        help='trial number of experiment spawned by hyperparameter tuner')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory to save output from training and logging')

    # Evals
    parser.add_argument('--eval-interval', type=int, default=None,
        help='number of global steps between evaluations')
    parser.add_argument('--eval-steps', type=int, default=None,
        help='number of sequential steps to evaluate for')
    parser.add_argument('--eval-num-envs', type=int, default=None,
        help='number of parallel environments in eval')
    parser.add_argument('--eval-env-kwargs', type=str, default=None,
        help='JSON dictionary with keyword arguments for the eval environment')
    parser.add_argument('--eval-processes', type=int, default=None,
                        help='The number of processes to use to collect evaluation data. The envs are split as equally as possible across the processes')
    parser.add_argument('--eval-capture-videos',type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If --eval-render-videos is set, videos will be recorded of the environments during evaluation')

    # Network architecture
    parser.add_argument('--d-model', type=int, default=64,
        help='the hidden size of the network layers')
    parser.add_argument('--n-head', type=int, default=1,
        help='the number of attention heads')
    parser.add_argument('--d-qk', type=int, default=64,
        help='the size queries and keys in action heads')
    parser.add_argument('--n-layer', type=int, default=1,
        help='the number of layers of the network')
    parser.add_argument('--pooling-op', type=str, default=None,
        help='if set, use pooling op instead of multi-head attention. Options: mean, max, meanmax')
    parser.add_argument('--translate', type=str, default=None,
        help='if set, translate positions to be centered on a given entity. Example: --translate=\'{"reference_entity": "SnakeHead", "position_features": ["x", "y"]}\'')
    parser.add_argument('--relpos-encoding', type=str, default=None,
        help='configuration for relative positional encoding. Example: --relpos-encoding=\'{"extent": [10, 10], "position_features": ["x", "y"]}\'')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--anneal-entropy', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="Toggle entropy coefficient annealing")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--microbatch-size', type=int, default=None,
        help='if set, use gradient accumulation to split up batches into smaller microbatches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    args = parser.parse_args(args=override_args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer: Any, std: float = np.sqrt(2), bias_const: float = 0.0) -> Any:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)


@dataclass
class RaggedBatchDict(Generic[ScalarType]):
    rb_cls: Type[RaggedBuffer[ScalarType]]
    buffers: Dict[str, RaggedBuffer[ScalarType]] = field(default_factory=dict)

    def extend(self, batch: Mapping[str, RaggedBuffer[ScalarType]]) -> None:
        for k, v in batch.items():
            if k not in self.buffers:
                self.buffers[k] = v
            else:
                self.buffers[k].extend(v)

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

    def __getitem__(
        self, index: npt.NDArray[np.int64]
    ) -> Dict[str, RaggedBuffer[ScalarType]]:
        return {k: v[index] for k, v in self.buffers.items()}


@dataclass
class RaggedActionDict:
    buffers: Dict[str, VecActionMask] = field(default_factory=dict)

    def extend(self, batch: Mapping[str, VecActionMask]) -> None:
        for k, v in batch.items():
            if k not in self.buffers:
                self.buffers[k] = v
            else:
                self.buffers[k].extend(v)

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

    def __getitem__(self, index: npt.NDArray[np.int64]) -> Dict[str, VecActionMask]:
        return {k: v[index] for k, v in self.buffers.items()}


def tensor_dict_to_ragged(
    rb_cls: Type[RaggedBuffer[ScalarType]],
    d: Dict[str, torch.Tensor],
    lengths: Dict[str, np.ndarray],
) -> Dict[str, RaggedBuffer[ScalarType]]:
    return {k: rb_cls.from_flattened(v.cpu().numpy(), lengths[k]) for k, v in d.items()}


class PPOActor(AutoActor):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        d_model: int = 64,
        n_head: int = 1,
        d_qk: int = 16,
        n_layer: int = 1,
        pooling_op: Optional[str] = None,
        feature_transforms: Optional[TranslatePositions] = None,
        relpos_encoding: Optional[RelposEncodingConfig] = None,
    ):
        auxiliary_heads = nn.ModuleDict(
            {"value": head_creator.create_value_head(d_model)}
        )
        super().__init__(
            obs_space,
            action_space,
            d_model,
            n_head,
            d_qk,
            auxiliary_heads,
            n_layer=n_layer,
            pooling_op=pooling_op,
            feature_transforms=feature_transforms,
            relpos_encoding=relpos_encoding,
        )

    def get_value(
        self, entities: Dict[str, RaggedBufferF32], tracer: Tracer
    ) -> torch.Tensor:
        return self.get_auxiliary_head(entities, "value", tracer)


class Rollout:
    def __init__(
        self,
        envs: VecEnv,
        obs_space: ObsSpace,
        action_space: Mapping[str, ActionSpace],
        agent: PPOActor,
        device: torch.device,
        tracer: Tracer,
    ) -> None:
        self.envs = envs
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device
        self.agent = agent
        self.tracer = tracer

        self.global_step = 0
        self.next_obs: Optional[VecObs] = None
        self.next_done: Optional[torch.Tensor] = None
        self.rewards = torch.zeros(0)
        self.dones = torch.zeros(0)
        self.values = torch.zeros(0)
        self.entities: RaggedBatchDict[np.float32] = RaggedBatchDict(RaggedBufferF32)
        self.action_masks = RaggedActionDict()
        self.actions = RaggedBatchDict(RaggedBufferI64)
        self.logprobs = RaggedBatchDict(RaggedBufferF32)

        self.rendered_frames: List[npt.NDArray[np.uint8]] = []
        self.rendered: Optional[npt.NDArray[np.uint8]] = None

    def run(
        self, steps: int, record_samples: bool, capture_videos: bool = False
    ) -> Tuple[VecObs, torch.Tensor, Dict[str, float]]:
        """
        Run the agent for a number of steps. Returns next_obs, next_done, and a dictionary of statistics.
        """
        if record_samples:
            if self.rewards.shape != (steps, len(self.envs)):
                self.rewards = torch.zeros((steps, len(self.envs))).to(self.device)
                self.dones = torch.zeros((steps, len(self.envs))).to(self.device)
                self.values = torch.zeros((steps, len(self.envs))).to(self.device)
            self.entities.clear()
            self.action_masks.clear()
            self.actions.clear()
            self.logprobs.clear()

        total_episodic_return = 0.0
        total_episodic_length = 0
        total_episodes = 0

        if self.next_obs is None or self.next_done is None:
            next_obs = self.envs.reset(self.obs_space)
            next_done = torch.zeros(len(self.envs)).to(self.device)
        else:
            next_obs = self.next_obs
            next_done = self.next_done

        if capture_videos:
            self.rendered_frames.append(self.envs.render(mode="rgb_array"))

        for step in range(steps):
            self.global_step += len(self.envs)

            if record_samples:
                self.entities.extend(next_obs.features)
                self.action_masks.extend(next_obs.action_masks)
                self.dones[step] = next_done

            with torch.no_grad(), self.tracer.span("forward"):
                (
                    action,
                    probs_tensor,
                    _,
                    actor_counts,
                    aux,
                    logits,
                ) = self.agent.get_action_and_auxiliary(
                    next_obs.features, next_obs.action_masks, tracer=self.tracer
                )
                logprob = tensor_dict_to_ragged(
                    RaggedBufferF32, probs_tensor, actor_counts
                )
            if record_samples:
                self.values[step] = aux["value"].flatten()
                self.actions.extend(action)
                self.logprobs.extend(logprob)

            if capture_videos:
                self.rendered_frames.append(self.envs.render(mode="rgb_array"))

            with self.tracer.span("step"):
                if isinstance(self.envs, SampleRecordingVecEnv):
                    if args.capture_logits:
                        ragged_logits: Optional[
                            Dict[str, RaggedBufferF32]
                        ] = tensor_dict_to_ragged(
                            RaggedBufferF32,
                            {k: v.squeeze(1) for k, v in logits.items()},
                            actor_counts,
                        )
                    else:
                        ragged_logits = None
                    next_obs = self.envs.act(
                        action, self.obs_space, logprob, ragged_logits
                    )
                else:
                    next_obs = self.envs.act(action, self.obs_space)

            if record_samples:
                with self.tracer.span("reward_done_to_device"):
                    self.rewards[step] = (
                        torch.tensor(next_obs.reward).to(self.device).view(-1)
                    )
                    next_done = torch.tensor(next_obs.done).to(self.device).view(-1)

            for eoei in next_obs.end_of_episode_info.values():
                total_episodic_return += eoei.total_reward
                total_episodic_length += eoei.length
                total_episodes += 1

        self.next_obs = next_obs
        self.next_done = next_done

        if capture_videos:
            self.rendered = np.stack(self.rendered_frames)

        metrics = {}
        if total_episodes > 0:
            avg_return = total_episodic_return / total_episodes
            avg_length = total_episodic_length / total_episodes
            metrics["charts/episodic_return"] = avg_return
            metrics["charts/episodic_length"] = avg_length
            metrics["charts/episodes"] = total_episodes
            metrics["meanrew"] = self.rewards.mean().item()
        return next_obs, next_done, metrics


def returns_and_advantages(
    agent: PPOActor,
    next_obs: VecObs,
    next_done: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gae: float,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    tracer: Tracer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # bootstrap value if not done
    next_value = agent.get_value(next_obs.features, tracer).reshape(1, -1)
    num_steps = values.size(0)
    if gae:
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return
        advantages = returns - values
    return returns, advantages


def run_eval(
    env_cls: Type[Environment],
    env_kwargs: Dict[str, Any],
    num_envs: int,
    processes: int,
    obs_space: ObsSpace,
    action_space: Mapping[str, ActionSpace],
    agent: PPOActor,
    device: torch.device,
    tracer: Tracer,
    writer: SummaryWriter,
    global_step: int,
    capture_videos: bool = False,
) -> None:
    # TODO: metrics are biased towards short episodes
    eval_envs: VecEnv
    if processes > 1:
        eval_envs = ParallelEnvList(
            env_cls,
            env_kwargs,
            num_envs,
            processes,
        )
    else:
        eval_envs = EnvList(
            env_cls, args.eval_env_kwargs or env_kwargs, args.eval_num_envs
        )
    eval_rollout = Rollout(
        eval_envs,
        obs_space=obs_space,
        action_space=action_space,
        agent=agent,
        device=device,
        tracer=tracer,
    )
    _, _, metrics = eval_rollout.run(
        args.eval_steps, record_samples=False, capture_videos=capture_videos
    )

    if capture_videos:
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


def train(args: argparse.Namespace) -> float:
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    config = vars(args)
    if os.path.exists("/xprun/info/config.ron"):
        import xprun  # type: ignore

        xp_info = xprun.current_xp()
        config["name"] = xp_info.xp_def.name
        config["base_name"] = xp_info.xp_def.base_name
        config["id"] = xp_info.id
        if args.trial is not None:
            args.seed = int(xp_info.xp_def.name.split("-")[-1])
        run_name = xp_info.xp_def.name
        out_dir: Optional[str] = os.path.join(
            "/mnt/xprun",
            xp_info.xp_def.project,
            xp_info.sanitized_name + "-" + xp_info.id,
        )
        Path(str(out_dir)).mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    data_path = Path(args.data_dir).absolute()
    data_path.mkdir(parents=True, exist_ok=True)
    data_dir = str(data_path)

    if args.track:
        import wandb

        config = vars(args)
        if os.path.exists("/xprun/info/config.ron"):
            import xprun

            xp_info = xprun.current_xp()
            config["name"] = xp_info.xp_def.name
            config["base_name"] = xp_info.xp_def.base_name
            config["id"] = xp_info.id
            if xp_info.xp_def.index is not None:
                args.seed = xp_info.xp_def.index
            run_name = xp_info.xp_def.name

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            dir=data_dir,
        )
    writer = SummaryWriter(os.path.join(data_dir, f"runs/{run_name}"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")
    tracer = Tracer(cuda=cuda)

    if args.gym_id in ENV_REGISTRY:
        env_cls = ENV_REGISTRY[args.gym_id]
    elif args.gym_id in GRIDDLY_ENVS:
        env_cls = create_env(**GRIDDLY_ENVS[args.gym_id])
    elif args.gym_id == "CodeCraft":
        env_cls = CodeCraftEnv
    else:
        raise KeyError(
            f"Unknown gym_id: {args.gym_id}\nAvailable environments: {list(ENV_REGISTRY.keys()) + list(GRIDDLY_ENVS.keys())}"
        )

    # env setup
    env_kwargs = json.loads(args.env_kwargs)
    if args.eval_env_kwargs is not None:
        eval_env_kwargs = json.loads(args.eval_env_kwargs)
    else:
        eval_env_kwargs = env_kwargs
    envs: VecEnv
    if args.gym_id == "CodeCraft":
        envs = CodeCraftVecEnv(args.num_envs, 0)
    elif args.processes > 1:
        envs = ParallelEnvList(env_cls, env_kwargs, args.num_envs, args.processes)
    else:
        envs = EnvList(env_cls, env_kwargs, args.num_envs)
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()
    if args.capture_samples:
        if out_dir is None:
            sample_file = args.capture_samples
        else:
            sample_file = os.path.join(out_dir, args.capture_samples)
        envs = SampleRecordingVecEnv(envs, sample_file)
    if args.translate:
        translate: Optional[TranslatePositions] = TranslatePositions(
            obs_space=obs_space, **json.loads(args.translate)
        )
    else:
        translate = None

    if args.relpos_encoding:
        relpos_encoding: Optional[RelposEncodingConfig] = RelposEncodingConfig(
            d_head=args.d_model // args.n_head,
            obs_space=obs_space,
            **json.loads(args.relpos_encoding),
        )
    else:
        relpos_encoding = None

    agent = PPOActor(
        obs_space,
        action_space,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        pooling_op=args.pooling_op,
        feature_transforms=translate,
        relpos_encoding=relpos_encoding,
    ).to(device)
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )
    if args.track:
        wandb.watch(agent)

    num_updates = args.total_timesteps // args.batch_size

    rollout = Rollout(
        envs,
        obs_space=obs_space,
        action_space=action_space,
        agent=agent,
        device=device,
        tracer=tracer,
    )

    if args.eval_interval is not None:
        next_eval_step: Optional[int] = 0
    else:
        next_eval_step = None

    start_time = time.time()
    for update in range(1, num_updates + 1):

        if next_eval_step is not None and rollout.global_step >= next_eval_step:
            next_eval_step += args.eval_interval

            # If eval processes is no set, we just use the same number of processes as the train loop
            eval_processes = args.eval_processes
            if not isinstance(eval_processes, int):
                eval_processes = args.processes

            run_eval(
                env_cls,
                eval_env_kwargs,
                args.eval_num_envs,
                eval_processes,
                obs_space,
                action_space,
                agent,
                device,
                tracer,
                writer,
                rollout.global_step,
                args.eval_capture_videos,
            )

        tracer.start("update")
        if (
            args.max_train_time is not None
            and time.time() - start_time >= args.max_train_time
        ):
            print("Max train time reached, stopping training.")
            break

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            if args.max_train_time is not None:
                frac = min(
                    frac, max(0, 1.0 - (time.time() - start_time) / args.max_train_time)
                )
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        tracer.start("rollout")

        next_obs, next_done, metrics = rollout.run(args.num_steps, record_samples=True)
        for name, value in metrics.items():
            writer.add_scalar(name, value, rollout.global_step)
        print(
            f"global_step={rollout.global_step} {'  '.join(f'{name}={value}' for name, value in metrics.items())}"
        )

        values = rollout.values
        actions = rollout.actions
        entities = rollout.entities
        action_masks = rollout.action_masks
        logprobs = rollout.logprobs
        global_step = rollout.global_step

        with torch.no_grad(), tracer.span("advantages"):
            returns, advantages = returns_and_advantages(
                agent,
                next_obs,
                next_done,
                rollout.rewards,
                rollout.dones,
                values,
                args.gae,
                args.gamma,
                args.gae_lambda,
                device,
                tracer,
            )

        # flatten the batch
        with tracer.span("flatten"):
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

        tracer.end("rollout")

        # Optimizaing the policy and value network
        tracer.start("optimize")
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                microbatch_size = (
                    args.microbatch_size
                    if args.microbatch_size is not None
                    else args.minibatch_size
                )

                optimizer.zero_grad()
                for _start in range(start, end, microbatch_size):
                    _end = _start + microbatch_size
                    mb_inds = b_inds[_start:_end]

                    b_entities = entities[mb_inds]
                    b_action_masks = action_masks[mb_inds]
                    b_logprobs = logprobs[mb_inds]
                    b_actions = actions[mb_inds]

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
                            b_action_masks,
                            prev_actions=b_actions,
                            tracer=tracer,
                        )
                        newvalue = aux["value"]

                    with tracer.span("ratio"):
                        logratio = {
                            k: newlogprob[k]
                            - torch.tensor(b_logprobs[k].as_array()).to(device)
                            for k in newlogprob.keys()
                        }
                        ratio = {k: l.exp() for k, l in logratio.items()}

                    with torch.no_grad(), tracer.span("kl"):
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        # TODO: mean across everything rather than nested mean? or do summation over different actions?
                        approx_kl = torch.tensor(
                            [
                                ((_ratio - 1) - _logratio).mean()
                                for (_ratio, _logratio) in zip(
                                    ratio.values(), logratio.values()
                                )
                            ]
                        ).mean()
                        clipfracs += [
                            torch.tensor(
                                [
                                    ((_ratio - 1.0).abs() > args.clip_coef)
                                    .float()
                                    .mean()
                                    .item()
                                    for _ratio in ratio.values()
                                ]
                            ).mean()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        assert (
                            len(mb_advantages) > 1
                        ), "Can't normalize advantages with minibatch size 1"
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # TODO: we can elide the index op and get better performance when there is exactly one actor per action
                    # TODO: we can reuse the mb_advantages across all actions that have the same number of actors
                    # TODO: what's the correct way of combining loss from multiple actions/actors on the same timestep? should we split the advantages across actions/actors?
                    with tracer.span("broadcast_advantages"):
                        # Brodcast the advantage value from each timestep to all actors/actions on that timestep
                        bc_mb_advantages = {
                            action_name: mb_advantages[
                                torch.tensor(
                                    _b_logprobs.indices(dim=0).as_array().flatten()
                                ).to(device),
                            ]
                            for action_name, _b_logprobs in b_logprobs.items()
                        }

                    # Policy loss
                    with tracer.span("policy_loss"):
                        pg_loss1 = torch.cat(
                            [
                                -_advantages * _ratio.flatten()
                                for _advantages, _ratio in zip(
                                    bc_mb_advantages.values(), ratio.values()
                                )
                            ]
                        )
                        pg_loss2 = torch.cat(
                            [
                                -_advantages
                                * torch.clamp(
                                    _ratio.flatten(),
                                    1 - args.clip_coef,
                                    1 + args.clip_coef,
                                )
                                for _advantages, _ratio in zip(
                                    bc_mb_advantages.values(), ratio.values()
                                )
                            ]
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    with tracer.span("value_loss"):
                        newvalue = newvalue.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # TODO: what's correct way of combining entropy loss from multiple actions/actors on the same timestep?
                    if args.anneal_entropy:
                        frac = 1.0 - (update - 1.0) / num_updates
                        if args.max_train_time is not None:
                            frac = min(
                                frac,
                                max(
                                    0,
                                    1.0
                                    - (time.time() - start_time) / args.max_train_time,
                                ),
                            )
                        ent_coef = frac * args.ent_coef
                    else:
                        ent_coef = args.ent_coef
                    entropy_loss = torch.cat([e for e in entropy.values()]).mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef
                    loss *= microbatch_size / args.minibatch_size

                    with tracer.span("backward"):
                        loss.backward()
                gradnorm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        tracer.end("optimize")

        tracer.start("metrics")
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
        for action_name, space in action_space.items():
            if isinstance(space, CategoricalActionSpace):
                _actions = actions.buffers[action_name].as_array().flatten()
                if len(_actions) > 0:
                    for i, label in enumerate(space.choices):
                        writer.add_scalar(
                            "actions/{}/{}".format(action_name, label),
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

    if args.eval_interval is not None:
        run_eval(
            env_cls,
            eval_env_kwargs,
            args.eval_num_envs,
            args.processes,
            obs_space,
            action_space,
            agent,
            device,
            tracer,
            writer,
            rollout.global_step,
        )

    envs.close()
    writer.close()

    return rollout.rewards.mean().item()


if __name__ == "__main__":
    args = parse_args()
    train(args)
