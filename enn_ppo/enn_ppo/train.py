# adapted from https://github.com/vwxyzjn/cleanrl
import argparse
from dataclasses import dataclass, field
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
)
import json

import numpy as np
import numpy.typing as npt
from entity_gym.environment import (
    ActionSpace,
    CategoricalAction,
    CategoricalActionSpace,
    EnvList,
    ObsSpace,
)
from entity_gym.envs import ENV_REGISTRY
from enn_zoo.griddly import GRIDDLY_ENVS, create_env
from enn_ppo.sample_recorder import SampleRecorder
from enn_ppo.simple_trace import Tracer
from rogue_net.actor import AutoActor
from rogue_net import head_creator
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
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
        help='total timesteps of the experiments')
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
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--capture-samples', type=str, default=None,
        help='if set, write the samples to this file')
    parser.add_argument('--max-log-frequency', type=int, default=None,
        help='if set, print episods stats at most every `max-log-frequency` timsteps')
    
    # Network architecture
    parser.add_argument('--hidden-size', type=int, default=64,
        help='the hidden size of the network layers')
    parser.add_argument('--n-layer', type=int, default=1,
        help='the number of layers of the network')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
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
        n_layer: int = 1,
    ):
        auxiliary_heads = nn.ModuleDict(
            {"value": head_creator.create_value_head(d_model)}
        )
        super().__init__(
            obs_space, action_space, d_model, auxiliary_heads, n_layer=n_layer
        )

    def get_value(
        self, entities: Dict[str, RaggedBufferF32], tracer: Tracer
    ) -> torch.Tensor:
        return self.get_auxiliary_head(entities, "value", tracer)


def train(args: argparse.Namespace) -> float:
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    tracer = Tracer()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.gym_id in ENV_REGISTRY:
        env_cls = ENV_REGISTRY[args.gym_id]
    elif args.gym_id in GRIDDLY_ENVS:
        path, level = GRIDDLY_ENVS[args.gym_id]
        env_cls = create_env(yaml_file=path, level=level)
    else:
        raise KeyError(
            f"Unknown gym_id: {args.gym_id}\nAvailable environments: {list(ENV_REGISTRY.keys()) + list(GRIDDLY_ENVS.keys())}"
        )

    # env setup
    env_kwargs = json.loads(args.env_kwargs)
    envs = EnvList([env_cls(**env_kwargs) for _ in range(args.num_envs)])  # type: ignore
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()
    if args.capture_samples:
        sample_recorder = SampleRecorder(args.capture_samples, action_space, obs_space)

    agent = PPOActor(
        obs_space, action_space, d_model=args.hidden_size, n_layer=args.n_layer
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    episodes = list(range(args.num_envs))
    curr_step = [0] * args.num_envs
    next_episode = args.num_envs
    last_log_step = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(obs_space)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    entities: RaggedBatchDict[np.float32] = RaggedBatchDict(RaggedBufferF32)
    action_masks = RaggedBatchDict(RaggedBufferI64)
    actions = RaggedBatchDict(RaggedBufferI64)
    logprobs = RaggedBatchDict(RaggedBufferF32)

    for update in range(1, num_updates + 1):
        tracer.start("update")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        tracer.start("rollout")
        entities.clear()
        action_masks.clear()
        actions.clear()
        logprobs.clear()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

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
            if args.capture_samples:
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
                    _action_dict = {}
                    for action_name, ragged_action_buffer in action.items():
                        mask = next_obs.action_masks[action_name][i]
                        _acts = ragged_action_buffer[i]
                        actor_action = [
                            (ids[actor_idx], _act)
                            for actor_idx, _act in zip(
                                mask.as_array().reshape(-1),
                                _acts.as_array().reshape(-1),
                            )
                        ]
                        _action_dict[action_name] = CategoricalAction(actor_action)
                    _actions.append(_action_dict)

            # TRY NOT TO MODIFY: execute the game and log data.
            with tracer.span("step"):
                next_obs = envs.act(_actions, obs_space)
            with tracer.span("reward_done_to_device"):
                rewards[step] = torch.tensor(next_obs.reward).to(device).view(-1)
                next_done = torch.tensor(next_obs.done).to(device).view(-1)

            for env_idx, eoei in enumerate(next_obs.end_of_episode_info.values()):
                if (
                    args.max_log_frequency is None
                    or args.max_log_frequency < global_step - last_log_step
                ):
                    print(
                        f"global_step={global_step + env_idx}, episodic_return={eoei.total_reward}"
                    )
                    last_log_step = global_step + env_idx
                if random.randint(0, 1000) == 0:
                    writer.add_scalar(
                        "charts/episodic_return",
                        eoei.total_reward,
                        global_step + env_idx,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        eoei.length,
                        global_step + env_idx,
                    )

            # TODO: reenable
            """
            if args.capture_samples:
                for i, o in enumerate(next_obs):
                    if o.done:
                        episodes[i] = next_episode
                        next_episode += 1
                        curr_step[i] = 0
                    else:
                        curr_step[i] += 1
            """

        # bootstrap value if not done
        with torch.no_grad(), tracer.span("advantages"):
            next_value = agent.get_value(next_obs.entities, tracer).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
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
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
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
                                _ratio.flatten(), 1 - args.clip_coef, 1 + args.clip_coef
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
                entropy_loss = torch.cat([e for e in entropy.values()]).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
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
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/gradnorm", gradnorm, global_step)
        writer.add_scalar("meanrew", rewards.mean().item(), global_step)
        for action_name, space in action_space.items():
            assert isinstance(space, CategoricalActionSpace)
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
    if args.capture_samples:
        sample_recorder.close()
    writer.close()

    return rewards.mean().item()


if __name__ == "__main__":
    args = parse_args()
    train(args)
