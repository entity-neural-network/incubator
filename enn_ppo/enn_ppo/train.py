# adapted from https://github.com/vwxyzjn/cleanrl
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from entity_gym.environment import (
    Action,
    ActionSpace,
    CategoricalAction,
    CategoricalActionSpace,
    EnvList,
    ObsSpace,
    Observation,
)
from entity_gym.envs import ENV_REGISTRY
from enn_ppo.sample_recorder import SampleRecorder, Sample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch_scatter


def parse_args(override_args: Optional[List[str]] = None) -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MoveToOrigin",
        help='the id of the gym environment')
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
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--capture-samples', type=str, default=None,
        help='if set, write the samples to this file')
    
    # Network architecture
    parser.add_argument('--hidden-size', type=int, default=64,
        help='the hidden size of the network layers')

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


class InputNorm(nn.Module):
    """
    Computes a running mean/variance of input features and performs normalization.
    Adapted from https://www.johndcook.com/blog/standard_deviation/
    """

    # Pretend that `count` is a float to make MyPy happy
    count: float

    def __init__(self, num_features: int, cliprange: float = 5) -> None:
        super(InputNorm, self).__init__()

        self.cliprange = cliprange
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("squares_sum", torch.zeros(num_features))
        self.fp16 = False
        self._stddev: Optional[torch.Tensor] = None
        self._dirty = True

    def update(self, input: torch.Tensor) -> None:
        self._dirty = True
        dbatch, dfeat = input.size()

        count = input.numel() / dfeat
        if count == 0:
            return
        mean = input.mean(dim=0)
        square_sum = ((input - mean) * (input - mean)).sum(dim=0)
        if self.count == 0:
            self.count += count
            self.mean = mean
            self.squares_sum = square_sum
        else:
            # This does not follow directly Welford's method since it is a batched update
            # Instead we consider computing the statistics of two sets, A="current set so far" B="current batch"
            # See Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979), "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.", Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
            delta = mean - self.mean
            self.mean += delta * count / (count + self.count)
            self.squares_sum += square_sum + torch.square(
                delta
            ) * count * self.count / (count + self.count)
            self.count += count

    def forward(
        self, input: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.training:
                self.update(input)
            if self.count > 1:
                input = (input - self.mean) / self.stddev()
            input = torch.clamp(input, -self.cliprange, self.cliprange)

        return input.half() if self.fp16 else input

    def enable_fp16(self) -> None:
        # Convert buffers back to fp32, fp16 has insufficient precision and runs into overflow on squares_sum
        self.float()
        self.fp16 = True

    def stddev(self) -> torch.Tensor:
        if self._dirty or self._stddev is None:
            sd = torch.sqrt(self.squares_sum / (self.count - 1))
            sd[sd == 0] = 1
            self._stddev = sd
            self._dirty = False
        return self._stddev


class Agent(nn.Module):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        d_model: int = 64,
    ):
        super(Agent, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.d_model = d_model
        self.embedding = nn.ModuleDict(
            {
                name: nn.Sequential(
                    InputNorm(len(entity.features)),
                    nn.Linear(len(entity.features), d_model),
                    nn.ReLU(),
                    nn.LayerNorm(d_model),
                )
                for name, entity in obs_space.entities.items()
            }
        )
        self.backbone = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),)
        action_heads = {}
        for name, space in action_space.items():
            assert isinstance(space, CategoricalActionSpace)
            action_heads[name] = nn.Linear(d_model, len(space.choices))
        self.action_heads = nn.ModuleDict(action_heads)
        self.value_head = nn.Linear(d_model, 1)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def batch_and_embed(
        self, obs: List[Observation]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Example:
        entities in obs 0: [A0, A0, A0, B0]
        entities in obs 1: [A1, B1, B1]
        entities in obs 2: [A2, A2]

        `x` is a flattened tensor of entity embeddings sorted first by entity type, then by batch index:
        [A0, A0, A0, A1, A2, A2, B0, B1, B1]

        `index_map` translates the index of entities sorted first by batch index then by entity type to their index in `x`:
        [0, 1, 2, 6, 3, 7, 8, 4, 5]

        `batch_index` gives the batch index of each entity in `x`:
        [0, 0, 0, 1, 2, 2, 0, 1, 1]

        `lengths` gives the number of entities in each observation:
        [4, 3, 2]
        """
        entity_embeds = []
        index_offsets = {}
        index_offset = 0
        for entity, embedding in self.embedding.items():
            batch = torch.cat(
                [torch.tensor(o.entities[entity], dtype=torch.float32) for o in obs],
            ).to(self.device())
            entity_embeds.append(embedding(batch))
            index_offsets[entity] = index_offset
            index_offset += batch.size(0)
        x = torch.cat(entity_embeds)
        index_map = []
        batch_index = []
        lengths = []
        for i, o in enumerate(obs):
            lengths.append(0)
            for entity in self.obs_space.entities.keys():
                count = len(o.entities[entity])
                index_map.append(
                    torch.arange(index_offsets[entity], index_offsets[entity] + count)
                )
                batch_index.append(torch.full((count,), i, dtype=torch.int64))
                index_offsets[entity] += count
                lengths[-1] += count
        x = self.backbone(x)

        return (
            x,
            torch.cat(index_map).to(self.device()),
            torch.cat(batch_index).to(self.device()),
            torch.tensor(lengths).to(self.device()),
        )

    def get_value(self, x: List[Observation]) -> torch.Tensor:
        embeddings, _, batch_index, _ = self.batch_and_embed(x)
        pooled = torch_scatter.scatter(
            src=embeddings, dim=0, index=batch_index, reduce="mean"
        )
        return self.value_head(pooled)  # type: ignore

    def get_action_and_value(
        self,
        obs: List[Observation],
        prev_actions: Optional[List[Dict[str, torch.Tensor]]] = None
        # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    ) -> Tuple[
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
        torch.Tensor,
    ]:
        actions = {}
        probs = {}
        entropies = {}
        x, index_map, batch_index, lengths = self.batch_and_embed(obs)
        index_offsets = (lengths.cumsum(0) - lengths[0]).cpu().numpy()
        actor_counts = {}
        for action_name, action_head in self.action_heads.items():
            actor_counts[action_name] = [
                len(o.action_masks[action_name].actors) for o in obs
            ]
            actors = torch.cat(
                [
                    torch.tensor(o.action_masks[action_name].actors + offset)
                    for (o, offset) in zip(obs, index_offsets)
                ]
            ).to(self.device())
            actor_embeds = x[index_map[actors]]
            logits = action_head(actor_embeds)
            dist = Categorical(logits=logits)
            if prev_actions is None:
                action = dist.sample()
            else:
                action = torch.cat([a[action_name] for a in prev_actions])
            actions[action_name] = action
            probs[action_name] = dist.log_prob(action)
            entropies[action_name] = dist.entropy()

        pooled = torch_scatter.scatter(src=x, dim=0, index=batch_index, reduce="mean")
        values = self.value_head(pooled)

        if prev_actions is None:
            unbatched_actions: List[Dict[str, torch.Tensor]] = [{} for _ in obs]
        unbatched_probs: List[Dict[str, torch.Tensor]] = [{} for _ in obs]
        unbatched_entropies: List[Dict[str, torch.Tensor]] = [{} for _ in obs]
        for action_name, ragged_batch_action_tensor in actions.items():
            if prev_actions is None:
                for action_dict, a in zip(
                    unbatched_actions,
                    torch.split(ragged_batch_action_tensor, actor_counts[action_name]),
                ):
                    action_dict[action_name] = a
            ragged_batch_probs_tensor = probs[action_name]
            for probs_dict, p in zip(
                unbatched_probs,
                torch.split(ragged_batch_probs_tensor, actor_counts[action_name]),
            ):
                probs_dict[action_name] = p
            ragged_batch_entropies_tensor = entropies[action_name]
            for entropies_dict, e in zip(
                unbatched_entropies,
                torch.split(ragged_batch_entropies_tensor, actor_counts[action_name]),
            ):
                entropies_dict[action_name] = e

        return (
            prev_actions or unbatched_actions,
            unbatched_probs,
            unbatched_entropies,
            values,
        )


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_cls = ENV_REGISTRY[args.gym_id]
    # env setup
    envs = EnvList([env_cls() for _ in range(args.num_envs)])
    obs_space = env_cls.obs_space()
    action_space = env_cls.action_space()
    if args.capture_samples:
        sample_recorder = SampleRecorder(args.capture_samples, action_space, obs_space)

    agent = Agent(obs_space, action_space, d_model=args.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    episodes = list(range(args.num_envs))
    curr_step = [0] * args.num_envs
    next_episode = args.num_envs

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(obs_space)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        obs = []
        actions = []
        logprobs = []

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs.append(next_obs)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions.append(action)
            logprobs.append(logprob)
            if args.capture_samples:
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

            # Join all actions with corresponding `EntityID`s
            _actions = []
            for _obs, action_dict in zip(next_obs, action):
                _action_dict = {}
                for action_name, action_tensor in action_dict.items():
                    mask = _obs.action_masks[action_name]
                    actor_action = [
                        (_obs.ids[actor_idx], _act)
                        for actor_idx, _act in zip(
                            mask.actors, action_tensor.cpu().numpy()
                        )
                    ]
                    _action_dict[action_name] = CategoricalAction(actor_action)
                _actions.append(_action_dict)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs = envs.act(_actions, obs_space)
            rewards[step] = (
                torch.tensor([o.reward for o in next_obs]).to(device).view(-1)
            )
            next_done = torch.tensor([o.done for o in next_obs]).to(device).view(-1)

            for i, o in enumerate(next_obs):
                if o.end_of_episode_info is not None:
                    print(
                        f"global_step={global_step}, episodic_return={o.end_of_episode_info.total_reward}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        o.end_of_episode_info.total_reward,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        o.end_of_episode_info.length,
                        global_step,
                    )
                    break
            if args.capture_samples:
                for i, o in enumerate(next_obs):
                    if o.done:
                        episodes[i] = next_episode
                        next_episode += 1
                        curr_step[i] = 0
                    else:
                        curr_step[i] += 1

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_obs = [o for _obs in obs for o in _obs]
        b_logprobs = [l for _logprobs in logprobs for l in _logprobs]
        b_actions = [a for _actions in actions for a in _actions]
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        def dictcat(x: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat(list(x.values()))

        # Optimizaing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    [b_obs[i] for i in mb_inds], [b_actions[i] for i in mb_inds]
                )
                logratio = [
                    dictcat(newlogprob[i]) - dictcat(b_logprobs[mb_inds[i]])
                    for i in range(len(mb_inds))
                ]
                ratio = [l.exp() for l in logratio]

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    # TODO: mean across everything rather than nested mean?
                    approx_kl = torch.tensor(
                        [
                            ((_ratio - 1) - _logratio).mean()
                            for (_ratio, _logratio) in zip(ratio, logratio)
                        ]
                    ).mean()
                    clipfracs += [
                        torch.tensor(
                            [
                                ((_ratio - 1.0).abs() > args.clip_coef)
                                .float()
                                .mean()
                                .item()
                                for _ratio in ratio
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

                # Policy loss
                pg_loss1 = torch.cat(
                    [
                        -_mb_advantages * _ratio
                        for (_mb_advantages, _ratio) in zip(mb_advantages, ratio)
                    ]
                )
                pg_loss2 = torch.cat(
                    [
                        -_mb_advantages
                        * torch.clamp(_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        for (_mb_advantages, _ratio) in zip(mb_advantages, ratio)
                    ]
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = torch.cat([dictcat(e) for e in entropy]).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                gradnorm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

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
            choices = torch.cat([a[action_name] for a in b_actions]).cpu().numpy()
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

    # envs.close()
    if args.capture_samples:
        sample_recorder.close()
    writer.close()

    return rewards.mean().item()


if __name__ == "__main__":
    args = parse_args()
    train(args)
