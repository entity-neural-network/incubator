from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import ragged_buffer
import torch
from ragged_buffer import RaggedBufferBool, RaggedBufferF32, RaggedBufferI64

from enn_ppo.agent import PPOAgent
from entity_gym.environment import *
from entity_gym.ragged_dict import RaggedActionDict, RaggedBatchDict
from entity_gym.serialization.sample_recorder import SampleRecordingVecEnv
from entity_gym.simple_trace import Tracer
from rogue_net.rogue_net import tensor_dict_to_ragged


class Rollout:
    def __init__(
        self,
        envs: VecEnv,
        obs_space: ObsSpace,
        action_space: Mapping[str, ActionSpace],
        agent: Union[PPOAgent, List[Tuple[npt.NDArray[np.int64], PPOAgent]]],
        device: torch.device,
        tracer: Tracer,
        value_function: Optional[PPOAgent] = None,
    ) -> None:
        self.envs = envs
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device
        self.agent = agent
        self.value_function = value_function
        self.tracer = tracer

        self.global_step = 0
        self.next_obs: Optional[VecObs] = None
        self.next_done: Optional[torch.Tensor] = None
        self.rewards = torch.zeros(0)
        self.dones = torch.zeros(0)
        self.values = torch.zeros(0)
        self.entities: RaggedBatchDict[np.float32] = RaggedBatchDict(RaggedBufferF32)
        self.visible: RaggedBatchDict[np.bool_] = RaggedBatchDict(RaggedBufferBool)
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
            self.visible.clear()
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
                # TODO: this breaks if entity missing on some steps, need merge full VecObs
                self.entities.extend(next_obs.features)
                self.visible.extend(next_obs.visible)
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
                                name: visible[env_indices]
                                for name, visible in next_obs.visible.items()
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
                        next_obs.features,
                        next_obs.visible,
                        next_obs.action_masks,
                        tracer=self.tracer,
                    )
                    logprob = tensor_dict_to_ragged(
                        RaggedBufferF32, probs_tensor, actor_counts
                    )
            if record_samples:
                if self.value_function is None:
                    value = aux["value"]
                else:
                    # TODO: can ignore `visible` here, allow for full attention across all entities
                    value = self.value_function.get_auxiliary_head(
                        next_obs.features, next_obs.visible, "value", tracer=self.tracer
                    )

                # Need to detach here because bug in pytorch that otherwise causes spurious autograd errors and memory leaks when dedicated value function network is used.
                # possibly same cause as this: https://github.com/pytorch/pytorch/issues/71495
                self.values[step] = value.detach().flatten()
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
                metrics[f"metrics/{k}/avg"] = v / total_episodes
        return next_obs, next_done, metrics
