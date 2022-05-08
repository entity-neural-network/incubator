from typing import Mapping, Tuple

import torch
from ragged_buffer import RaggedBufferF32

from enn_ppo.config import PPOConfig
from entity_gym.environment.environment import ActionName
from entity_gym.simple_trace import Tracer


def ppo_loss(
    cfg: PPOConfig,
    newlogprob: Mapping[ActionName, torch.Tensor],
    oldlogprob: Mapping[ActionName, RaggedBufferF32],
    advantages: torch.Tensor,
    device: torch.device,
    tracer: Tracer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with tracer.span("ratio"):
        logratio = {
            k: newlogprob[k]
            - torch.tensor(oldlogprob[k].as_array(), device=device).squeeze(-1)
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
                for (_ratio, _logratio) in zip(ratio.values(), logratio.values())
            ]
        ).mean()
        clipfrac = torch.tensor(
            [
                ((_ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                for _ratio in ratio.values()
            ]
        ).mean()

    # TODO: not invariant to microbatch size, should be normalizing full batch or minibatch instead
    if cfg.norm_adv:
        assert len(advantages) > 1, "Can't normalize advantages with minibatch size 1"
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # TODO: we can elide the index op and get better performance when there is exactly one actor per action
    # TODO: we can reuse the mb_advantages across all actions that have the same number of actors
    # TODO: what's the correct way of combining loss from multiple actions/actors on the same timestep? should we split the advantages across actions/actors?
    with tracer.span("broadcast_advantages"):
        # Broadcast the advantage value from each timestep to all actors/actions on that timestep
        bc_mb_advantages = {
            action_name: advantages[
                torch.tensor(
                    _b_logprobs.indices(dim=0).as_array().flatten(), device=device
                )
            ]
            for action_name, _b_logprobs in oldlogprob.items()
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
                    1 - cfg.clip_coef,
                    1 + cfg.clip_coef,
                )
                for _advantages, _ratio in zip(
                    bc_mb_advantages.values(), ratio.values()
                )
            ]
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    return pg_loss, clipfrac, approx_kl


def value_loss(
    cfg: PPOConfig,
    newvalue: torch.Tensor,
    returns: torch.Tensor,
    oldvalue: torch.Tensor,
    tracer: Tracer,
) -> torch.Tensor:
    with tracer.span("value_loss"):
        newvalue = newvalue.view(-1)
        if cfg.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = oldvalue + torch.clamp(
                newvalue - oldvalue,
                -cfg.clip_coef,
                cfg.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    return v_loss  # type: ignore
