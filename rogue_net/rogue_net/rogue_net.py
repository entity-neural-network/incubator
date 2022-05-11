import dataclasses
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch_scatter
from ragged_buffer import (
    RaggedBuffer,
    RaggedBufferBool,
    RaggedBufferF32,
    RaggedBufferI64,
)

from entity_gym.env import ActionSpace, ObsSpace, VecActionMask
from entity_gym.env.environment import (
    CategoricalActionSpace,
    Entity,
    GlobalCategoricalActionSpace,
    SelectEntityActionSpace,
)
from entity_gym.simple_trace import Tracer
from rogue_net.categorical_action_head import CategoricalActionHead
from rogue_net.embedding import EntityEmbedding
from rogue_net.ragged_tensor import RaggedTensor
from rogue_net.select_entity_action_head import PaddedSelectEntityActionHead
from rogue_net.transformer import Transformer, TransformerConfig
from rogue_net.translate_positions import TranslationConfig

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


@dataclass
class RogueNetConfig(TransformerConfig):
    """RogueNet network parameters.

    Attributes:
        d_qk: dimension of keys and queries in select-entity action heads
        translation: settings for transforming all position features to be centered on one entity
    """

    d_qk: int = 16
    translation: Optional[TranslationConfig] = None


class RogueNet(nn.Module):
    def __init__(
        self,
        cfg: RogueNetConfig,
        obs_space: ObsSpace,
        action_space: Dict[str, ActionSpace],
        regression_heads: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        global_features = obs_space.global_features
        _obs_space = dataclasses.replace(obs_space, global_features=[])
        if len(global_features) > 0:
            _obs_space.entities = {
                label: Entity(entity.features + global_features)
                for label, entity in _obs_space.entities.items()
            }
        if any(
            isinstance(a, GlobalCategoricalActionSpace) for a in action_space.values()
        ):
            _obs_space.entities["__global__"] = Entity(features=global_features)

        self.d_model = cfg.d_model
        self.action_space = action_space
        self.obs_space = obs_space
        self._obs_space = _obs_space
        self.embedding = EntityEmbedding(_obs_space, cfg.translation, cfg.d_model)
        self.backbone = Transformer(cfg, _obs_space)
        self.action_heads = create_action_heads(action_space, cfg.d_model, cfg.d_qk)
        self.auxiliary_heads = (
            nn.ModuleDict(
                {
                    name: regression_head(cfg.d_model, d_out)
                    for name, d_out in regression_heads.items()
                }
            )
            if regression_heads is not None
            else None
        )

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def batch_and_embed(
        self,
        entities: Mapping[str, RaggedBufferF32],
        visible: Mapping[str, RaggedBufferBool],
        tracer: Tracer,
    ) -> RaggedTensor:
        with tracer.span("embedding"):
            # Ensure consistent dictionary ordering
            entities = {
                name: entities[name]
                for name in list(self.obs_space.entities.keys()) + ["__global__"]
                if name in entities
            }
            (
                x,
                tbatch_index,
                index_map,
                tentities,
                tindex_map,
                entity_types,
                tlengths,
            ) = self.embedding(entities, tracer, self.device())

        with tracer.span("visibility_mask"):
            if len(visible) > 0:
                visibilities = []
                for etype, feats in entities.items():
                    if etype in visible:
                        visibilities.append(
                            torch.tensor(
                                visible[etype].as_array(), device=self.device()
                            ).view(-1)
                        )
                    else:
                        visibilities.append(
                            torch.ones(
                                feats.items(),
                                dtype=torch.bool,
                                device=self.device(),
                            )
                        )
                tvisible: Optional[torch.Tensor] = torch.cat(visibilities, dim=0)[
                    tindex_map
                ]
            else:
                tvisible = None

        with tracer.span("backbone"):
            x = self.backbone(
                x,
                tbatch_index,
                index_map,
                tentities,
                tindex_map,
                entity_types,
                tvisible,
            )

        return RaggedTensor(
            x,
            tbatch_index,
            tlengths,
        )

    def get_auxiliary_head(
        self,
        entities: Mapping[str, RaggedBufferF32],
        visible: Mapping[str, RaggedBufferBool],
        head_name: str,
        tracer: Tracer,
    ) -> torch.Tensor:
        x = self.batch_and_embed(entities, visible, tracer)
        pooled = torch_scatter.scatter(
            src=x.data, dim=0, index=x.batch_index, reduce="mean"
        )
        return self.auxiliary_heads[head_name](pooled)  # type: ignore

    def get_action_and_auxiliary(
        self,
        entities: Mapping[str, RaggedBufferF32],
        visible: Mapping[str, RaggedBufferBool],
        action_masks: Mapping[str, VecActionMask],
        tracer: Tracer,
        prev_actions: Optional[Dict[str, RaggedBufferI64]] = None,
    ) -> Tuple[
        Dict[str, RaggedBufferI64],  # actions
        Dict[str, torch.Tensor],  # chosen action probabilities
        Dict[str, torch.Tensor],  # entropy
        Dict[str, npt.NDArray[np.int64]],  # number of actors in each frame
        Dict[str, torch.Tensor],  # auxiliary head values
        Dict[str, torch.Tensor],  # full logits
    ]:
        actions = {}
        probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        with tracer.span("batch_and_embed"):
            x = self.batch_and_embed(entities, visible, tracer)

        tracer.start("action_heads")
        index_offsets = RaggedBufferI64.from_array(
            torch.cat([torch.tensor([0]).to(self.device()), x.lengths[:-1]])
            .cumsum(0)
            .cpu()
            .numpy()
            .reshape(-1, 1, 1)
        )
        actor_counts: Dict[str, np.ndarray] = {}
        for action_name, action_head in self.action_heads.items():
            action, count, logprob, entropy, logit = action_head(
                x,
                index_offsets,
                action_masks[action_name],
                prev_actions[action_name] if prev_actions is not None else None,
            )
            actor_counts[action_name] = count
            actions[action_name] = action
            probs[action_name] = logprob
            entropies[action_name] = entropy
            if logit is not None:
                logits[action_name] = logit

        tracer.end("action_heads")

        tracer.start("auxiliary_heads")
        if self.auxiliary_heads:
            pooled = torch.zeros(
                x.lengths.size(0), x.data.size(1), device=x.data.device
            )
            torch_scatter.scatter(
                src=x.data,
                dim=0,
                index=x.batch_index,
                reduce="mean",
                out=pooled,
            )
            auxiliary_values = {
                name: module(pooled) for name, module in self.auxiliary_heads.items()
            }
        else:
            auxiliary_values = {}
        tracer.end("auxiliary_heads")

        return (
            prev_actions
            or tensor_dict_to_ragged(RaggedBufferI64, actions, actor_counts),
            probs,
            entropies,
            actor_counts,
            auxiliary_values,
            logits,
        )


def regression_head(d_model: int, d_out: int) -> nn.Module:
    projection = nn.Linear(d_model, d_out)
    projection.weight.data.fill_(0.0)
    projection.bias.data.fill_(0.0)
    return projection


def create_action_heads(
    action_space: Dict[str, ActionSpace], d_model: int, d_qk: int
) -> nn.ModuleDict:
    action_heads: Dict[str, nn.Module] = {}
    for name, space in action_space.items():
        if isinstance(space, CategoricalActionSpace) or isinstance(
            space, GlobalCategoricalActionSpace
        ):
            action_heads[name] = CategoricalActionHead(d_model, len(space))
        elif isinstance(space, SelectEntityActionSpace):
            action_heads[name] = PaddedSelectEntityActionHead(d_model, d_qk)
        else:
            raise ValueError(f"Unknown action space {space}")
    return nn.ModuleDict(action_heads)
