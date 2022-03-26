from typing import Any, Mapping, Optional, Type
from entity_gym.environment.environment import ActionType, Environment, ObsSpace
from entity_gym.environment.vec_env import Metric, VecEnv, VecObs

import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferI64


class AddMetricsWrapper(VecEnv):
    def __init__(
        self, env: VecEnv, filter: Optional[npt.NDArray[np.bool8]] = None
    ) -> None:
        """
        Wrap a VecEnv to track and add metrics for (episodic) rewards and episode lengths.

        Args:
            env: The VecEnv to wrap.
            filter: A boolean array of length len(env) indicating which environments to
                track metrics for. If filter[i] is True, then the metrics for the i-th
                environment will be tracked.
        """
        self.env = env
        self.total_reward = np.zeros(len(env), dtype=np.float32)
        self.total_steps = np.zeros(len(env), dtype=np.int64)
        self.filter = filter or np.ones(len(env), dtype=np.bool8)

    def env_cls(self) -> Type[Environment]:
        return self.env.env_cls()

    def reset(self, obs_config: ObsSpace) -> VecObs:
        return self.track_metrics(self.env.reset(obs_config))

    def act(
        self, actions: Mapping[ActionType, RaggedBufferI64], obs_filter: ObsSpace
    ) -> VecObs:
        return self.track_metrics(self.env.act(actions, obs_filter))

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        return self.env.render(**kwargs)

    def __len__(self) -> int:
        return len(self.env)

    def close(self) -> None:
        self.env.close()

    def track_metrics(self, obs: VecObs) -> VecObs:
        self.total_reward += obs.reward
        self.total_steps += 1
        episodic_reward = Metric()
        episodic_length = Metric()
        for i in np.arange(len(self))[obs.done & self.filter]:
            episodic_reward.push(self.total_reward[i])
            episodic_length.push(self.total_steps[i])
            self.total_reward[i] = 0.0
            self.total_steps[i] = 0
        obs.metrics["episodic/reward"] = episodic_reward
        obs.metrics["episodic/length"] = episodic_length
        obs.metrics["reward"] = Metric(
            sum=obs.reward.sum(),
            count=obs.reward.size,
            min=obs.reward.min(),
            max=obs.reward.max(),
        )
        return obs
