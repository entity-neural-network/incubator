# adapted from https://github.com/vwxyzjn/cleanrl
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Any, Dict, List, Type

import numpy as np
from entity_gym.environment import (
    ActionSpace,
    CategoricalAction,
    DenseCategoricalActionMask,
    EnvList,
    Environment,
    ObsFilter,
    Observation,
)
from entity_gym.envs.cherry_pick import CherryPick
from entity_gym.envs.minefield import Minefield
from entity_gym.envs.move_to_origin import MoveToOrigin
from entity_gym.envs.multi_snake import MultiSnake
from entity_gym.envs.pick_matching_balls import PickMatchingBalls
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


ENV_MAPPING: Dict[str, Type[Environment]] = {
    "MoveToOrigin": MoveToOrigin,
    "CherryPick": CherryPick,
    "PickMatchingBalls": PickMatchingBalls,
    "Minefield": Minefield,
    "MultiSnake": MultiSnake,
}


def layer_init(layer: Any, std: float = np.sqrt(2), bias_const: float = 0.0) -> Any:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        obs_filter: ObsFilter,
        action_space: Dict[str, ActionSpace],
        d_model: int = 64,
    ):
        super(Agent, self).__init__()
        self.d_model = d_model
        self.embedding = nn.ModuleDict(
            {
                entity: nn.Linear(len(features), d_model)
                for entity, features in obs_filter.entity_to_feats.items()
            }
        )
        self.action_heads = nn.ModuleDict(
            {
                entity: nn.Linear(d_model, len(action_space[entity].choices))
                for entity in action_space
            }
        )
        self.value_head = nn.Linear(d_model, 1)

    def embed(self, obs: Observation) -> torch.Tensor:
        return torch.cat(
            [
                self.embedding[name](
                    torch.tensor(features.astype(np.float32)).to(
                        self.embedding[name].weight.device
                    )
                )
                for name, features in obs.entities.items()
            ]
        )

    def get_value(self, x: List[Observation]) -> torch.Tensor:
        values = []
        for obs in x:
            embeddings = self.embed(obs)
            values.append(self.value_head(embeddings).mean())
        return torch.tensor(values, device=self.value_head.weight.device)

    def get_action_and_value(self, x: List[Observation], prev_actions=None):
        actions = []
        probs = []
        entropies = []
        values = []
        for i, obs in enumerate(x):
            embeddings = self.embed(obs)
            _actions = {}
            _probs = []
            _entropies = []
            for action_name, mask in obs.action_masks.items():
                assert isinstance(mask, DenseCategoricalActionMask)
                selected_embeddings = embeddings[mask.actors, :]
                # TODO: apply mask
                logits = self.action_heads[action_name](selected_embeddings)
                dist = Categorical(logits=logits)
                # TODO: entity id, multiple entities
                if prev_actions is None:
                    action = dist.sample()
                    _actions[action_name] = CategoricalAction([(0, action.item())])
                else:
                    action = torch.tensor(
                        prev_actions[i][action_name].actions[0][1], device=logits.device
                    )
                _probs.append(dist.log_prob(action))
                _entropies.append(dist.entropy())
            actions.append(_actions)
            probs.append(torch.cat(_probs))
            entropies.append(torch.cat(_entropies))
            values.append(self.value_head(embeddings).mean())
        return (
            actions,
            probs,
            entropies,
            torch.tensor(values, device=self.value_head.weight.device),
        )


if __name__ == "__main__":
    args = parse_args()
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

    env_cls = ENV_MAPPING[args.gym_id]
    # env setup
    envs = EnvList([env_cls() for _ in range(args.num_envs)])
    obs_filter = env_cls.full_obs_filter()

    agent = Agent(obs_filter, env_cls.action_space()).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = []
    actions = []
    logprobs = []
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(obs_filter)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
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

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs = envs.act(action, obs_filter)
            rewards[step] = (
                torch.tensor([o.reward for o in next_obs]).to(device).view(-1)
            )
            next_done = torch.tensor([o.done for o in next_obs]).to(device).view(-1)

            # for item in info:
            #    if "episode" in item.keys():
            #        print(
            #            f"global_step={global_step}, episodic_return={item['episode']['r']}"
            #        )
            #        writer.add_scalar(
            #            "charts/episodic_return", item["episode"]["r"], global_step
            #        )
            #        writer.add_scalar(
            #            "charts/episodic_length", item["episode"]["l"], global_step
            #        )
            #        break

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
                    newlogprob[i] - b_logprobs[mb_inds[i]] for i in range(len(mb_inds))
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

                entropy_loss = torch.cat(entropy).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
