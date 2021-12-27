# adapted from https://github.com/vwxyzjn/cleanrl
import argparse
from dataclasses import asdict, dataclass, field
import os
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
    Type,
    TypeVar,
    Union,
)
import json
import hyperstate

import numpy as np
import numpy.typing as npt
from entity_gym.environment import (
    ActionMaskBatch,
    ActionSpace,
    CategoricalAction,
    CategoricalActionSpace,
    DenseSelectEntityActionMask,
    EnvList,
    ObsSpace,
    SelectEntityAction,
    SelectEntityActionMaskBatch,
    SelectEntityActionSpace,
)
from entity_gym.envs import ENV_REGISTRY
from enn_zoo.griddly import GRIDDLY_ENVS, create_env
from enn_ppo.sample_recorder import SampleRecorder
from enn_ppo.simple_trace import Tracer
from rogue_net.actor import AutoActor, NetworkOpts
from rogue_net import head_creator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--hps", nargs="+", help="Override hyperparameter value")
    return parser.parse_args()


@dataclass
class TaskConfig:
    id: str = "MoveToOrigin"
    """the id of the gym environment"""
    kwargs: str = "{}"
    """JSON dictionary with keyword arguments for the environment"""


@dataclass
class Config:
    net: NetworkOpts
    task: TaskConfig
    exp_name: str = field(
        default_factory=lambda: os.path.basename(__file__).rstrip(".py")
    )
    """the name of this experiment"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    seed: int = 1
    """seed of the experiment"""
    total_timesteps: int = 25000
    """total timesteps of the experiments"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "enn-ppo"
    """the wandb's project name"""
    wandb_entity: str = "entity-neural-network"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """wether to capture videos of the agent performances (check out `videos` folder)"""
    capture_samples: Optional[str] = None
    """if set, write the samples to this file"""
    # Algorithm specific arguments
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gae: bool = True
    """Use GAE for advantage computation"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles wheter or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy loss"""
    vf_coef: float = 0.5
    """coefficient of the value function loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


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
    buffers: Dict[str, ActionMaskBatch] = field(default_factory=dict)

    def extend(self, batch: Mapping[str, ActionMaskBatch]) -> None:
        for k, v in batch.items():
            if k not in self.buffers:
                self.buffers[k] = v
            else:
                self.buffers[k].extend(v)

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()

    def __getitem__(self, index: npt.NDArray[np.int64]) -> Dict[str, ActionMaskBatch]:
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
        opts: NetworkOpts,
    ):
        auxiliary_heads = nn.ModuleDict(
            {"value": head_creator.create_value_head(opts.d_model)}
        )
        super().__init__(obs_space, action_space, opts, auxiliary_heads)

    def get_value(
        self, entities: Dict[str, RaggedBufferF32], tracer: Tracer
    ) -> torch.Tensor:
        return self.get_auxiliary_head(entities, "value", tracer)


def train(args: argparse.Namespace) -> float:
    config = hyperstate.load(Config, path=args.config, overrides=args.hps)

    run_name = f"{config.task.id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=asdict(config),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in asdict(config).items()])),
    )

    tracer = Tracer()

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.cuda else "cpu"
    )

    if config.task.id in ENV_REGISTRY:
        env_cls = ENV_REGISTRY[config.task.id]
    elif config.task.id in GRIDDLY_ENVS:
        path, level = GRIDDLY_ENVS[config.task.id]
        env_cls = create_env(yaml_file=path, level=level)
    else:
        raise KeyError(
            f"Unknown gym_id: {config.task.id}\nAvailable environments: {list(ENV_REGISTRY.keys()) + list(GRIDDLY_ENVS.keys())}"
        )

    # env setup
    env_kwargs = json.loads(config.task.kwargs)
    envs = EnvList([env_cls(**env_kwargs) for _ in range(config.num_envs)])  # type: ignore
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()
    if config.capture_samples:
        sample_recorder = SampleRecorder(
            config.capture_samples, action_space, obs_space
        )

    agent = PPOActor(
        obs_space,
        action_space,
        config.net,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    episodes = list(range(config.num_envs))
    curr_step = [0] * config.num_envs
    next_episode = config.num_envs
    last_log_step = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(obs_space)
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size

    entities: RaggedBatchDict[np.float32] = RaggedBatchDict(RaggedBufferF32)
    action_masks = RaggedActionDict()
    actions = RaggedBatchDict(RaggedBufferI64)
    logprobs = RaggedBatchDict(RaggedBufferF32)

    for update in range(1, num_updates + 1):
        tracer.start("update")

        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        tracer.start("rollout")
        entities.clear()
        action_masks.clear()
        actions.clear()
        logprobs.clear()
        total_episodic_return = 0.0
        total_episodic_length = 0
        total_episodes = 0
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs

            entities.extend(next_obs.entities)
            action_masks.extend(next_obs.action_masks)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad(), tracer.span("forward"):
                (
                    action,
                    probs_tensor,
                    _,
                    actor_counts,
                    aux,
                ) = agent.get_action_and_auxiliary(
                    next_obs.entities, next_obs.action_masks, tracer=tracer
                )
                logprob = tensor_dict_to_ragged(
                    RaggedBufferF32, probs_tensor, actor_counts
                )
                values[step] = aux["value"].flatten()
            actions.extend(action)
            logprobs.extend(logprob)
            if config.capture_samples:
                with tracer.span("record_samples"):
                    # TODO: fix
                    """
                    for i, o in enumerate(next_obs):
                        sample_recorder.record(
                            Sample(
                                entities=o.entities,
                                action_masks={
                                    n: a.actors for n, a in o.action_masks.items()
                                },
                                # TODO: capture full logprobs, not just chosen action
                                probabilities={
                                    n: l.cpu().numpy() for n, l in logprob[i].items()
                                },
                                # TODO: actually want to capture returns, need to move after rollout
                                reward=o.reward,
                                step=curr_step[i],
                                episode=episodes[i],
                            )
                        )
                    """

            # Join all actions with corresponding `EntityID`s
            with tracer.span("join_actions"):
                _actions = []
                for i, ids in enumerate(next_obs.ids):
                    _action_dict: Dict[
                        str, Union[CategoricalAction, SelectEntityAction]
                    ] = {}
                    for action_name, ragged_action_buffer in action.items():
                        mask = next_obs.action_masks[action_name]
                        _acts = ragged_action_buffer[i]
                        _as = action_space[action_name]
                        if isinstance(_as, CategoricalActionSpace):
                            actor_action = [
                                (ids[actor_idx], _act)
                                for actor_idx, _act in zip(
                                    mask.actors[i].as_array().reshape(-1),
                                    _acts.as_array().reshape(-1),
                                )
                            ]
                            _action_dict[action_name] = CategoricalAction(actor_action)
                        elif isinstance(_as, SelectEntityActionSpace):
                            assert isinstance(
                                mask, SelectEntityActionMaskBatch
                            ), f"Expected SelectEntityActionMaskBatch, got {type(mask)}"
                            actees = mask.actees[i].as_array().flatten()
                            actor_action = [
                                (ids[actor_idx], ids[actees[actee_idx]])
                                for actor_idx, actee_idx in zip(
                                    mask.actors[i].as_array().reshape(-1),
                                    _acts.as_array().reshape(-1),
                                )
                            ]
                            _action_dict[action_name] = SelectEntityAction(actor_action)
                    _actions.append(_action_dict)

            # TRY NOT TO MODIFY: execute the game and log data.
            with tracer.span("step"):
                next_obs = envs.act(_actions, obs_space)
            with tracer.span("reward_done_to_device"):
                rewards[step] = torch.tensor(next_obs.reward).to(device).view(-1)
                next_done = torch.tensor(next_obs.done).to(device).view(-1)

            for eoei in next_obs.end_of_episode_info.values():
                total_episodic_return += eoei.total_reward
                total_episodic_length += eoei.length
                total_episodes += 1

            # TODO: reenable
            """
            if config.capture_samples:
                for i, o in enumerate(next_obs):
                    if o.done:
                        episodes[i] = next_episode
                        next_episode += 1
                        curr_step[i] = 0
                    else:
                        curr_step[i] += 1
            """

        if total_episodes > 0:
            avg_return = total_episodic_return / total_episodes
            avg_length = total_episodic_length / total_episodes
            writer.add_scalar(
                "charts/episodic_return",
                avg_return,
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_length",
                avg_length,
                global_step,
            )
            writer.add_scalar(
                "charts/episodes",
                total_episodes,
                global_step,
            )
            print(
                f"global_step={global_step}, episodic_return={avg_return}, episodic_length={avg_length}, episodes={total_episodes}"
            )

        # bootstrap value if not done
        with torch.no_grad(), tracer.span("advantages"):
            next_value = agent.get_value(next_obs.entities, tracer).reshape(1, -1)
            if config.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + config.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + config.gamma
                        * config.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = (
                        rewards[t] + config.gamma * nextnonterminal * next_return
                    )
                advantages = returns - values

        # flatten the batch
        with tracer.span("flatten"):
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

        tracer.end("rollout")

        def dictcat(x: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat(list(x.values()))

        # Optimizaing the policy and value network
        tracer.start("optimize")
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                with tracer.span("shuffle"):
                    end = start + config.minibatch_size
                    mb_inds = b_inds[start:end]

                    b_entities = entities[mb_inds]
                    b_action_masks = action_masks[mb_inds]
                    b_logprobs = logprobs[mb_inds]
                    b_actions = actions[mb_inds]

                with tracer.span("forward"):
                    _, newlogprob, entropy, _, aux = agent.get_action_and_auxiliary(
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
                                ((_ratio - 1.0).abs() > config.clip_coef)
                                .float()
                                .mean()
                                .item()
                                for _ratio in ratio.values()
                            ]
                        ).mean()
                    ]

                with tracer.span("advantages"):
                    mb_advantages = b_advantages[mb_inds]
                    if config.norm_adv:
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
                                1 - config.clip_coef,
                                1 + config.clip_coef,
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
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -config.clip_coef,
                            config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                with tracer.span("loss"):
                    # TODO: what's correct way of combining entropy loss from multiple actions/actors on the same timestep?
                    entropy_loss = torch.cat([e for e in entropy.values()]).mean()
                    loss = (
                        pg_loss
                        - config.ent_coef * entropy_loss
                        + v_loss * config.vf_coef
                    )

                with tracer.span("zero_grad"):
                    optimizer.zero_grad()
                with tracer.span("backward"):
                    loss.backward()
                with tracer.span("gradclip"):
                    gradnorm = nn.utils.clip_grad_norm_(
                        agent.parameters(), config.max_grad_norm
                    )
                with tracer.span("step"):
                    optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
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
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/gradnorm", gradnorm, global_step)
        writer.add_scalar("meanrew", rewards.mean().item(), global_step)
        for action_name, space in action_space.items():
            if isinstance(space, CategoricalActionSpace):
                choices = actions.buffers[action_name].as_array().flatten()
                for i, label in enumerate(space.choices):
                    writer.add_scalar(
                        "actions/{}/{}".format(action_name, label),
                        np.sum(choices == i).item() / len(choices),
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

    # envs.close()
    if config.capture_samples:
        sample_recorder.close()
    writer.close()

    return rewards.mean().item()


if __name__ == "__main__":
    args = parse_args()
    train(args)
