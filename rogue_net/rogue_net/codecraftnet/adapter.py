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
from ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBuffer


class CCNetAdapter:
    def __init__(self, device: str) -> None:
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
                feat_mineral_claims=True,
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

    def parameters(self) -> Any:
        return self.network.parameters()

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
        )
        globals = entities["ally"].as_array()[
            :, -self.network.obs_config.global_features() :
        ]
        obs[:, : oc.endglobals()] = torch.tensor(globals)
        allies = entities["ally"]
        for i in range(allies.size0()):
            obs[i, oc.endglobals() : oc.endallies()] = torch.tensor(
                allies[i].as_array()
            )
        minerals = entities["mineral"]
        for i in range(minerals.size0()):
            obs[i, oc.endenemies() : oc.endmins()] = torch.tensor(
                minerals[i].as_array()
            )

        __import__("ipdb").set_trace()
