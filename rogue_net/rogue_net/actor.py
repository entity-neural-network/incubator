from typing import Dict, List, Optional, Tuple

from entity_gym.environment import (
    ActionSpace,
    ObsSpace,
    Observation,
)
from enn_ppo.simple_trace import Tracer
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch_scatter

import rogue_net.head_creator as head_creator
import rogue_net.embedding_creator as embedding_creator


class Actor(nn.Module):
    def __init__(
        self,
        obs_space: ObsSpace,
        embedding: nn.ModuleDict,
        action_space: Dict[str, ActionSpace],
        backbone: nn.Module,
        action_heads: nn.ModuleDict,
        auxiliary_heads: Optional[nn.ModuleDict] = None,
    ):
        super(Actor, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.embedding = embedding
        self.backbone = backbone
        self.action_heads = action_heads
        self.auxiliary_heads = auxiliary_heads

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def batch_and_embed(
        self, obs: List[Observation],
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

    def get_auxiliary_head(self, x: List[Observation], head_name: str) -> torch.Tensor:
        embeddings, _, batch_index, _ = self.batch_and_embed(x)
        pooled = torch_scatter.scatter(
            src=embeddings, dim=0, index=batch_index, reduce="mean"
        )
        return self.auxiliary_heads[head_name](pooled)  # type: ignore

    def get_action_and_auxiliary(
        self,
        obs: List[Observation],
        prev_actions: Optional[List[Dict[str, torch.Tensor]]] = None,
        tracer: Optional[Tracer] = None,
    ) -> Tuple[
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
    ]:
        actions = {}
        probs = {}
        entropies = {}
        if tracer:
            tracer.start("batch_and_embed")
        x, index_map, batch_index, lengths = self.batch_and_embed(obs)
        if tracer:
            tracer.end("batch_and_embed")
            tracer.start("action_heads")
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
            # TODO: this could be generalized with the involvement of the action head perhaps?
            dist = Categorical(logits=logits)
            if prev_actions is None:
                action = dist.sample()
            else:
                action = torch.cat([a[action_name] for a in prev_actions])
            actions[action_name] = action
            probs[action_name] = dist.log_prob(action)
            entropies[action_name] = dist.entropy()
        if tracer:
            tracer.end("action_heads")

        if tracer:
            tracer.start("auxiliary_heads")
        if self.auxiliary_heads:
            pooled = torch_scatter.scatter(
                src=x, dim=0, index=batch_index, reduce="mean"
            )
            auxiliary_values = {
                name: module(pooled) for name, module in self.auxiliary_heads.items()
            }
        else:
            auxiliary_values = {}
        if tracer:
            tracer.end("auxiliary_heads")

        if tracer:
            tracer.start("unbatch")
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
        if tracer:
            tracer.end("unbatch")

        return (
            prev_actions or unbatched_actions,
            unbatched_probs,
            unbatched_entropies,
            auxiliary_values,
        )


class AutoActor(Actor):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        d_model: int,
        backbone: nn.Module,
        auxiliary_heads: Optional[nn.ModuleDict] = None,
    ):
        self.d_model = d_model
        super().__init__(
            obs_space,
            embedding_creator.create_embeddings(obs_space, d_model),
            action_space,
            backbone,
            head_creator.create_action_heads(action_space, d_model),
            auxiliary_heads=auxiliary_heads,
        )
