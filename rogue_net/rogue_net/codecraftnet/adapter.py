import numpy as np
import numpy.typing as npt
from typing import Any, Mapping, Optional, Dict, Tuple
from enn_ppo.simple_trace import Tracer
from entity_gym.environment.vec_env import VecActionMask
import torch
from rogue_net.codecraftnet.codecraftnet import (
    ObsConfig,
    PolicyConfig,
    TransformerPolicy8HS,
)
import torch.nn as nn
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBuffer


class CCNetAdapter(nn.Module):
    def __init__(self, device: str) -> None:
        super(CCNetAdapter, self).__init__()
        self.network = TransformerPolicy8HS(
            PolicyConfig(
                agents=1,
                ally_enemy_same=False,
                d_agent=256,
                d_item=128,
                dff_ratio=2,
                dropout=0,
                item_ff=True,
                item_item_attn_layers=0,
                map_embed_offset=False,
                mc_kernel_size=3,
                nally=1,
                nconstant=0,
                nearby_map=False,
                nenemy=0,
                nhead=8,
                nmineral=10,
                norm="layernorm",
                ntile=0,
                small_init_pi=False,
                zero_init_vf=True,
            ),
            ObsConfig(
                allies=1,
                feat_abstime=True,
                feat_construction_progress=True,
                feat_dist_to_wall=True,
                feat_is_visible=True,
                feat_last_seen=False,
                feat_map_size=True,
                feat_mineral_claims=False,
                feat_unit_count=True,
                harvest_action=False,
                lock_build_action=False,
                obs_enemies=0,
                obs_keep_abspos=True,
                obs_map_tiles=0,
                obs_minerals=10,
                use_privileged=False,
            ),
            naction=8,
        ).to(device)
        self.device = device

    def get_action_and_auxiliary(
        self,
        entities: Mapping[str, RaggedBufferF32],
        action_masks: Mapping[str, VecActionMask],
        tracer: Tracer,
        prev_actions: Optional[Dict[str, RaggedBufferI64]] = None,
    ) -> Tuple[
        Dict[str, RaggedBufferI64],  # actions
        Dict[str, torch.Tensor],  # action probabilities
        Dict[str, torch.Tensor],  # entropy
        Dict[str, npt.NDArray[np.int64]],  # number of actors in each frame
        Dict[str, torch.Tensor],  # auxiliary head values
        Dict[str, torch.Tensor],  # full logits
    ]:
        # Undo ragged buffers
        oc = self.network.obs_config
        obs = torch.zeros(
            (entities["ally"].size0(), oc.stride()),
        ).to(self.device)
        allies = entities["ally"]
        globals = allies.as_array()[:, -self.network.obs_config.global_features() :]
        obs[:, : oc.endglobals()] = torch.tensor(globals)
        for i in range(allies.size0()):
            obs[
                i, oc.endglobals() : oc.endglobals() + allies.size1(i) * oc.dstride()
            ] = torch.tensor(
                allies[i].as_array()[:, : -self.network.obs_config.global_features()]
            ).view(
                -1
            )
        minerals = entities["mineral"]
        for i in range(minerals.size0()):
            obs[
                i, oc.endenemies() : oc.endenemies() + minerals.size1(i) * oc.mstride()
            ] = torch.tensor(minerals[i].as_array()).view(-1)

        actions, logprobs, entropy, values, probs = self.network.evaluate(
            obs,
            torch.tensor(action_masks["act"].mask.as_array())
            .view(allies.size0(), oc.allies, 8)
            .to(self.device)
            if "act" in action_masks
            else torch.ones((allies.size0(), oc.allies, 8), dtype=torch.bool).to(
                self.device
            ),
            privileged_obs=None,
            prev_actions=torch.tensor(prev_actions["act"].as_array()).to(self.device) if prev_actions is not None else None,
        )
        # print("ENTITIES", entities)
        # TODO: variable number of actors
        return (
            {"act": RaggedBufferI64.from_array(actions.unsqueeze(-1).cpu().numpy())},
            {"act": logprobs},
            {"act": entropy.unsqueeze(-1)},
            {"act": allies.size1()},
            {"value": values.unsqueeze(-1)},
            None,
        )

    def get_value(
        self, entities: Dict[str, RaggedBufferF32], tracer: Tracer
    ) -> torch.Tensor:
        return self.get_action_and_auxiliary(
            entities, action_masks={}, tracer=tracer, prev_actions=None
        )[4]["value"]
