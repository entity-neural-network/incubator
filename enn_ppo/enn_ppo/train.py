# adapted from https://github.com/vwxyzjn/cleanrl
import argparse
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import random
import time
from distutils.util import strtobool
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import json
from entity_gym.environment import *
from entity_gym.environment.validator import validated_env
from entity_gym.ragged_dict import RaggedActionDict, RaggedBatchDict

import hyperstate

import numpy as np
import numpy.typing as npt
from entity_gym.examples import ENV_REGISTRY
from enn_zoo.griddly import GRIDDLY_ENVS, create_env
from enn_zoo.microrts import GymMicrorts
from enn_zoo.codecraft.cc_vec_env import codecraft_env_class, CodeCraftVecEnv

from enn_zoo.codecraft.codecraftnet.adapter import CCNetAdapter

from entity_gym.serialization import SampleRecordingVecEnv
from enn_ppo.simple_trace import Tracer
from rogue_net.actor import AutoActor
from rogue_net import head_creator

from rogue_net.translate_positions import TranslatePositions, TranslationConfig
from rogue_net.transformer import TransformerConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import ragged_buffer
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBuffer


@dataclass
class EvalConfig:
    """Evaluation settings

    Attributes:
        interval: number of global steps between evaluations
        on_step_0: whether to run eval on step 0
        steps: number of sequential steps to evaluate for
        num_envs: number of parallel environments in eval
        env_kwargs: JSON dictionary with keyword arguments for the eval environment
        capture_videos: if --eval-render-videos is set, videos will be recorded of the environments during evaluation
        capture_samples: if set, write the samples from evals to this file
        capture_samples_subsample: only persist every nth sample, chosen randomly
        capture_logits: if --eval-capture-samples is set, record full logits of the agent
        codecraft_eval: if toggled, run evals with CodeCraft environment
        codecraft_eval_opponent: path to CodeCraft policy to evaluate against
        codecraft_only_opponent: run only the opponent, not the agent
    """

    interval: int
    steps: int
    num_envs: int
    processes: Optional[int] = None
    env_kwargs: str = "{}"
    capture_videos: bool = False
    capture_samples: str = ""
    capture_logits: bool = True
    codecraft_eval: bool = False
    capture_samples_subsample: int = 1
    codecraft_eval_opponent: Optional[str] = ""
    run_on_first_step: bool = True
    codecraft_only_opponent: bool = False


@dataclass
class EnvConfig:
    """Environment settings.

    Attributes:
        kwargs: JSON dictionary with keyword arguments for the environment.
        num_envs: The number of game environments.
        num_steps: The number of steps to run in each environment per policy rollout.
        processes: The number of processes to use to collect env data. The envs are split as equally as possible across the processes.
        id: The id of the environment.
        validate: Check that all observations returned by the environment are valid. Disable for better performance.
    """

    kwargs: str = "{}"
    num_envs: int = 4
    num_steps: int = 128
    processes: int = 1
    id: str = "MoveToOrigin"
    validate: bool = True


@dataclass
class PPOConfig:
    """Proximal Policy Optimization settings.

    Attributes:
        gae: whether to use GAE for advantage computation
        gamma: the discount factor gamma
        gae_lambda: the lambda for the general advantage estimation
        norm_adv: whether to normalize advantages
        clip_coef: the surrogate clipping coefficient
        clip_vloss: whether or not to use a clipped loss for the value function, as per the paper
        ent_coef: coefficient of the entropy
        vf_coef: coefficient of the value function
        target_kl: the target KL divergence threshold
        anneal_entropy: whether to anneal the entropy coefficient
    """

    gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    target_kl: Optional[float] = None
    anneal_entropy: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer settings.

    Attributes:
        lr: the learning rate of the optimizer
        bs: the batch size of the optimizer
        micro_bs: if set, use gradient accumulation to split up batches into smaller microbatches
        weight_decay: the weight decay of the optimizer
        anneal_lr: whether to anneal the learning rate
        update_epochs: the K epochs to update the policy
        max_grad_norm: the maximum norm for the gradient clipping
    """

    lr: float = 2.5e-4
    bs: int = 128
    weight_decay: float = 0.0
    micro_bs: Optional[int] = None
    anneal_lr: bool = True
    update_epochs: int = 3
    max_grad_norm: float = 2.0


@dataclass
class ExperimentConfig:
    """Experiment settings.

    Attributes:
        net: policy network configuration
        d_dk: dimension of keys and queries in select-entity action heads
        translation: settings for transforming all position features to be centered on one entity
        relpos_encoding: settings for relative position encoding
        codecraft_net: if toggled, use the DeepCodeCraft policy network instead of RogueNet (only works with CodeCraft environment)
        name: the name of the experiment
        seed: seed of the experiment
        total_timesteps: total timesteps of the experiments
        max_train_time: train for at most this many seconds
        torch_deterministic: if toggled, `torch.backends.cudnn.deterministic=False`
        cuda: if toggled, cuda will be enabled by default
        track: if toggled, this experiment will be tracked with Weights and Biases
        wandb_project_name: the wandb's project name
        wandb_entity: the entity (team) of wandb's project
        capture_samples: if set, write the samples to this file
        capture_logits: If --capture-samples is set, record full logits of the agent
        capture_samples_subsample: only persist every nth sample, chosen randomly
        trial: trial number of experiment spawned by hyperparameter tuner
        data_dir: Directory to save output from training and logging
    """

    env: EnvConfig
    net: TransformerConfig
    optim: OptimizerConfig
    ppo: PPOConfig
    translation: Optional[TranslationConfig] = None
    d_qk: int = 64
    eval: Optional[EvalConfig] = None
    codecraft_net: bool = False

    name: str = field(default_factory=lambda: os.path.basename(__file__).rstrip(".py"))
    seed: int = 1
    total_timesteps: int = 25000
    max_train_time: Optional[int] = None
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "enn-ppo"
    wandb_entity: str = "entity-neural-network"
    capture_samples: Optional[str] = None
    capture_logits: bool = False
    capture_samples_subsample: int = 1
    trial: Optional[int] = None
    data_dir: str = "."


def layer_init(layer: Any, std: float = np.sqrt(2), bias_const: float = 0.0) -> Any:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)


def tensor_dict_to_ragged(
    rb_cls: Type[RaggedBuffer[ScalarType]],
    d: Dict[str, torch.Tensor],
    lengths: Dict[str, np.ndarray],
) -> Dict[str, RaggedBuffer[ScalarType]]:
    result = {}
    for k, v in d.items():
        flattened = v.cpu().numpy()
        if flattened.ndim == 1:
            flattened = flattened.reshape(-1, 1)
        result[k] = rb_cls.from_flattened(flattened, lengths[k])
    return result


class PPOActor(AutoActor):
    def __init__(
        self,
        tf: TransformerConfig,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        d_qk: int = 16,
        feature_transforms: Optional[TranslationConfig] = None,
    ):
        auxiliary_heads = nn.ModuleDict(
            {"value": head_creator.create_value_head(tf.d_model)}
        )
        super().__init__(
            tf,
            obs_space,
            action_space,
            d_qk,
            auxiliary_heads,
            feature_transforms=TranslatePositions(feature_transforms, obs_space)
            if feature_transforms
            else None,
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
        agent: Union[PPOActor, List[Tuple[npt.NDArray[np.int64], PPOActor]]],
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
        self,
        steps: int,
        record_samples: bool,
        capture_videos: bool = False,
        capture_logits: bool = False,
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
        if isinstance(self.agent, list):
            allindices = np.concatenate([indices for indices, _ in self.agent])
            invindex = np.zeros_like(allindices, dtype=np.int64)
            for i, index in enumerate(allindices):
                invindex[index] = i
        else:
            invindex = np.array([], dtype=np.int64)

        total_episodic_return = 0.0
        total_episodic_length = 0
        total_metrics = {}
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
                if isinstance(self.agent, list):
                    actions = []
                    for env_indices, agent in self.agent:
                        a = agent.get_action_and_auxiliary(
                            {
                                name: feats[env_indices]
                                for name, feats in next_obs.features.items()
                            },
                            {
                                name: mask[env_indices]
                                for name, mask in next_obs.action_masks.items()
                            },
                            self.tracer,
                        )[0]
                        actions.append((env_indices, a))
                    action = {}
                    for name in self.action_space.keys():
                        action[name] = ragged_buffer.cat([a[1][name] for a in actions])[
                            invindex
                        ]
                else:
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
                    if capture_logits:
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

            if isinstance(self.agent, list):
                end_of_episode_infos = []
                for i in self.agent[0][0]:
                    if i in next_obs.end_of_episode_info:
                        end_of_episode_infos.append(next_obs.end_of_episode_info[i])
            else:
                end_of_episode_infos = list(next_obs.end_of_episode_info.values())
            for eoei in end_of_episode_infos:
                total_episodic_return += eoei.total_reward
                total_episodic_length += eoei.length
                total_episodes += 1
                if eoei.metrics is not None:
                    for k, v in eoei.metrics.items():
                        if k not in total_metrics:
                            total_metrics[k] = v
                        else:
                            total_metrics[k] += v

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
            for k, v in total_metrics.items():
                metrics[f"metrics/{k}"] = v / total_episodes
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
    cfg: EvalConfig,
    env_cfg: EnvConfig,
    env_cls: Type[Environment],
    obs_space: ObsSpace,
    action_space: Mapping[str, ActionSpace],
    agent: PPOActor,
    device: torch.device,
    tracer: Tracer,
    writer: SummaryWriter,
    global_step: int,
) -> None:
    # TODO: metrics are biased towards short episodes

    if cfg.env_kwargs is None:
        env_kwargs = json.loads(env_cfg.env_kwargs)
    else:
        env_kwargs = json.loads(cfg.env_kwargs)

    eval_envs: VecEnv
    processes = cfg.processes or env_cfg.processes
    if cfg.codecraft_eval:
        eval_envs = CodeCraftVecEnv(
            cfg.num_envs, stagger=False, symmetric=True, **env_kwargs
        )
    elif processes > 1:
        eval_envs = ParallelEnvList(
            env_cls,
            env_kwargs,
            cfg.num_envs,
            processes,
        )
    else:
        eval_envs = EnvList(env_cls, env_kwargs, cfg.num_envs)

    if cfg.codecraft_eval:
        if cfg.codecraft_only_opponent:
            agents: Union[PPOActor, List[Tuple[npt.NDArray[np.int64], PPOActor]]] = CCNetAdapter(device, load_from=eval_opponent)  # type: ignore
        else:
            if cfg.codecraft_eval_opponent is None:
                opponent = PPOActor(
                    TransformerConfig(),
                    obs_space,
                    dict(action_space),
                ).to(device)
            else:
                opponent = CCNetAdapter(device, load_from=eval_opponent)  # type: ignore
            agents = [
                (np.array([2 * i for i in range(cfg.num_envs // 2)]), agent),
                (
                    np.array([2 * i + 1 for i in range(cfg.num_envs // 2)]),
                    opponent,
                ),
            ]
    else:
        agents = agent
    if cfg.capture_samples:
        eval_envs = SampleRecordingVecEnv(
            eval_envs, cfg.capture_samples, cfg.capture_samples_subsample
        )
    eval_rollout = Rollout(
        eval_envs,
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
    eval_envs.close()


def train(cfg: ExperimentConfig) -> float:
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

    env_kwargs = json.loads(cfg.env.kwargs)

    if cfg.env.id in ENV_REGISTRY:
        env_cls = ENV_REGISTRY[cfg.env.id]
    elif cfg.env.id in GRIDDLY_ENVS:
        env_cls = create_env(**GRIDDLY_ENVS[cfg.env.id])
    elif cfg.env.id == "CodeCraft":
        env_cls = codecraft_env_class(env_kwargs.get("objective", "ALLIED_WEALTH"))
    elif cfg.env.id == "GymMicrorts":
        env_cls = GymMicrorts
    else:
        raise KeyError(
            f"Unknown gym_id: {cfg.env.id}\nAvailable environments: {list(ENV_REGISTRY.keys()) + list(GRIDDLY_ENVS.keys())}"
        )

    # env setup
    envs: VecEnv
    if cfg.env.validate:
        env_cls = validated_env(env_cls)
    if cfg.env.id == "CodeCraft":
        envs = CodeCraftVecEnv(cfg.env.num_envs, **env_kwargs)
    elif cfg.env.processes > 1:
        envs = ParallelEnvList(env_cls, env_kwargs, cfg.env.num_envs, cfg.env.processes)
    else:
        envs = EnvList(env_cls, env_kwargs, cfg.env.num_envs)
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()
    if cfg.capture_samples:
        if out_dir is None:
            sample_file = cfg.capture_samples
        else:
            sample_file = os.path.join(out_dir, cfg.capture_samples)
        envs = SampleRecordingVecEnv(envs, sample_file, cfg.capture_samples_subsample)

    if not cfg.codecraft_net:
        agent = PPOActor(
            cfg.net,
            obs_space,
            action_space,
            feature_transforms=cfg.translation,
        ).to(device)
    else:
        agent = CCNetAdapter(device)  # type: ignore

    optimizer = optim.AdamW(
        agent.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=1e-5,
    )
    if cfg.track:
        wandb.watch(agent)

    num_updates = cfg.total_timesteps // (cfg.env.num_envs * cfg.env.num_steps)

    rollout = Rollout(
        envs,
        obs_space=obs_space,
        action_space=action_space,
        agent=agent,
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
            run_eval(
                cfg.eval,
                cfg.env,
                env_cls,
                obs_space,
                action_space,
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

        tracer.start("rollout")

        next_obs, next_done, metrics = rollout.run(
            cfg.env.num_steps, record_samples=True, capture_logits=cfg.capture_logits
        )
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
                cfg.ppo.gae,
                cfg.ppo.gamma,
                cfg.ppo.gae_lambda,
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
        frames = cfg.env.num_envs * cfg.env.num_steps
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
                            - torch.tensor(b_logprobs[k].as_array())
                            .squeeze(-1)
                            .to(device)
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
                                    ((_ratio - 1.0).abs() > cfg.ppo.clip_coef)
                                    .float()
                                    .mean()
                                    .item()
                                    for _ratio in ratio.values()
                                ]
                            ).mean()
                        ]

                    # TODO: not invariant to microbatch size, should be normalizing full batch or minibatch instead
                    mb_advantages = b_advantages[mb_inds]  # type: ignore
                    if cfg.ppo.norm_adv:
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
                                    1 - cfg.ppo.clip_coef,
                                    1 + cfg.ppo.clip_coef,
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
                        if cfg.ppo.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2  # type: ignore
                            v_clipped = b_values[mb_inds] + torch.clamp(  # type: ignore
                                newvalue - b_values[mb_inds],  # type: ignore
                                -cfg.ppo.clip_coef,
                                cfg.ppo.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2  # type: ignore
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()  # type: ignore

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

            if cfg.ppo.target_kl is not None:
                if approx_kl > cfg.ppo.target_kl:
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

    if cfg.eval is not None:
        _run_eval()

    envs.close()
    writer.close()

    return rollout.rewards.mean().item()


@hyperstate.command(ExperimentConfig)
def main(cfg: ExperimentConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
