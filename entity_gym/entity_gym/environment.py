import multiprocessing as mp
import multiprocessing.connection as conn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    Callable,
    Generator,
    overload,
)

import cloudpickle
import numpy as np
import numpy.typing as npt
from ragged_buffer import RaggedBufferF32, RaggedBufferI64


@dataclass
class CategoricalActionSpace:
    choices: List[str]


@dataclass
class SelectEntityActionSpace:
    pass


ActionSpace = Union[CategoricalActionSpace, SelectEntityActionSpace]


@dataclass
class ActionMask(ABC):
    """
    Base class for action masks that specify what agents can perform a particular action.
    """

    actors: npt.NDArray[np.int64]
    """
    The indices of the entities that can perform the action.
    """


@dataclass
class DenseCategoricalActionMask(ActionMask):
    """
    Action mask for categorical action that specifies which agents can perform the action,
    and includes a dense mask that further contraints the choices available to each agent.
    """

    mask: Optional[np.ndarray] = None
    """
    A boolean array of shape (len(actors), len(choices)). If mask[i, j] is True, then
    agent i can perform action j.
    """


@dataclass
class DenseSelectEntityActionMask(ActionMask):
    """
    Action mask for select entity action that specifies which agents can perform the action,
    and includes a dense mask that further contraints what other entities can be selected by
    each actor.
    """

    actees: npt.NDArray[np.int64]
    mask: Optional[np.ndarray] = None
    """
    An boolean array of shape (len(actors), len(entities)). If mask[i, j] is True, then
    agent i can select entity j.
    """


EntityID = Any


@dataclass
class EpisodeStats:
    length: int
    total_reward: float


@dataclass
class Observation:
    entities: Dict[str, np.ndarray]
    """Maps each entity type to an array with the features for each observed entity of that type."""
    ids: Sequence[EntityID]
    """
    Maps each entity index to an opaque identifier used by the environment to
    identify that entity.
    """
    action_masks: Mapping[str, ActionMask]
    """Maps each action to an action mask."""
    reward: float
    done: bool
    end_of_episode_info: Optional[EpisodeStats] = None


@dataclass
class CategoricalActionMaskBatch:
    actors: RaggedBufferI64

    def push(self, mask: Any) -> None:
        assert isinstance(mask, DenseCategoricalActionMask)
        self.actors.push(mask.actors.reshape(-1, 1))

    @overload
    def __getitem__(self, i: int) -> RaggedBufferI64:
        ...

    @overload
    def __getitem__(self, i: npt.NDArray[np.int64]) -> "CategoricalActionMaskBatch":
        ...

    def __getitem__(
        self, i: Union[int, npt.NDArray[np.int64]]
    ) -> Union["CategoricalActionMaskBatch", RaggedBufferI64]:
        if isinstance(i, int):
            return self.actors[i]
        else:
            return CategoricalActionMaskBatch(self.actors[i])

    def extend(self, other: Any) -> None:
        assert isinstance(
            other, CategoricalActionMaskBatch
        ), f"Expected CategoricalActionMaskBatch, got {type(other)}"
        self.actors.extend(other.actors)

    def clear(self) -> None:
        self.actors.clear()


@dataclass
class SelectEntityActionMaskBatch:
    actors: RaggedBufferI64
    actees: RaggedBufferI64

    def push(self, mask: Any) -> None:
        assert isinstance(
            mask, DenseSelectEntityActionMask
        ), f"Expected DenseSelectEntityActionMask, got {type(mask)}"
        self.actors.push(mask.actors.reshape(-1, 1))
        self.actees.push(mask.actees.reshape(-1, 1))

    @overload
    def __getitem__(self, i: int) -> RaggedBufferI64:
        ...

    @overload
    def __getitem__(self, i: npt.NDArray[np.int64]) -> "SelectEntityActionMaskBatch":
        ...

    def __getitem__(
        self, i: Union[int, npt.NDArray[np.int64]]
    ) -> Union["SelectEntityActionMaskBatch", RaggedBufferI64]:
        if isinstance(i, int):
            return self.actors[i]
        else:
            return SelectEntityActionMaskBatch(self.actors[i], self.actees[i])

    def extend(self, other: Any) -> None:
        assert isinstance(
            other, SelectEntityActionMaskBatch
        ), f"Expected SelectEntityActionMaskBatch, got {type(other)}"
        self.actors.extend(other.actors)
        self.actees.extend(other.actees)

    def clear(self) -> None:
        self.actors.clear()
        self.actees.clear()


ActionMaskBatch = Union[CategoricalActionMaskBatch, SelectEntityActionMaskBatch]


@dataclass
class ObsBatch:
    entities: Dict[str, RaggedBufferF32]
    ids: Sequence[Sequence[EntityID]]
    action_masks: Mapping[str, ActionMaskBatch]
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]
    end_of_episode_info: Dict[int, EpisodeStats]


def batch_obs(obs: List[Observation]) -> ObsBatch:
    """
    Converts a list of observations into a batch of observations.
    """
    entities = {}
    ids = []
    action_masks: Dict[
        str, Union[CategoricalActionMaskBatch, SelectEntityActionMaskBatch]
    ] = {}
    reward = []
    done = []
    end_of_episode_info = {}
    for o in obs:
        for k, feats in o.entities.items():
            if k not in entities:
                entities[k] = RaggedBufferF32(feats.shape[-1])
            entities[k].push(feats)
        ids.append(o.ids)
        for k, mask in o.action_masks.items():
            if isinstance(mask, DenseCategoricalActionMask):
                if k not in action_masks:
                    action_masks[k] = CategoricalActionMaskBatch(RaggedBufferI64(1))
            elif isinstance(mask, DenseSelectEntityActionMask):
                if k not in action_masks:
                    action_masks[k] = SelectEntityActionMaskBatch(
                        RaggedBufferI64(1), RaggedBufferI64(1)
                    )
            action_masks[k].push(mask)

        reward.append(o.reward)
        done.append(o.done)
        if o.end_of_episode_info:
            end_of_episode_info[len(ids) - 1] = o.end_of_episode_info
    return ObsBatch(
        entities,
        ids,
        action_masks,
        np.array(reward),
        np.array(done),
        end_of_episode_info,
    )


@dataclass
class Entity:
    features: List[str]


@dataclass
class ObsSpace:
    entities: Dict[str, Entity]


@dataclass
class CategoricalAction:
    # TODO: figure out best representation
    actions: List[Tuple[EntityID, int]]
    # actions: np.ndarray
    """
    Maps each actor to the index of the chosen action.
    Given `Observation` obs and `ActionMask` mask, the `EntityID`s of the corresponding
    actors are given as `obs.ids[mask.actors]`.
    """


@dataclass
class SelectEntityAction:
    actions: List[Tuple[EntityID, EntityID]]
    """Maps each actor to the entity they selected."""


Action = Union[CategoricalAction, SelectEntityAction]


class Environment(ABC):
    """
    Abstraction over reinforcement learning environments with observations based on structured lists of entities.

    As a simple hack to support basic multi-agent environments with parallel observations and actions,
    methods may return lists of observations and accept lists of actions.
    This should be replaced by a more general multi-agent environment interface in the future.
    """

    @classmethod
    @abstractmethod
    def obs_space(cls) -> ObsSpace:
        """
        Returns a dictionary mapping the name of observable entities to their type.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def action_space(cls) -> Dict[str, ActionSpace]:
        """
        Returns a dictionary mapping the name of actions to their action space.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def _act(self, action: Mapping[str, Action]) -> Observation:
        """
        Performs the given action and returns the resulting observation.

        Args:
            action: Maps the name of each action type to the action to perform.
        """
        raise NotImplementedError

    def reset(self, obs_filter: ObsSpace) -> Observation:
        return self.__class__.filter_obs(self._reset(), obs_filter)

    def act(self, action: Mapping[str, Action], obs_filter: ObsSpace) -> Observation:
        return self.__class__.filter_obs(self._act(action), obs_filter)

    def close(self) -> None:
        raise NotImplementedError

    @classmethod
    def filter_obs(cls, obs: Observation, obs_filter: ObsSpace) -> Observation:
        selectors = cls._compile_feature_filter(obs_filter)
        entities = {
            entity_name: entity_features[:, selectors[entity_name]].reshape(
                entity_features.shape[0], len(selectors[entity_name])
            )
            for entity_name, entity_features in obs.entities.items()
        }
        return Observation(
            entities,
            obs.ids,
            obs.action_masks,
            obs.reward,
            obs.done,
            obs.end_of_episode_info,
        )

    @classmethod
    def _compile_feature_filter(cls, obs_space: ObsSpace) -> Dict[str, np.ndarray]:
        obs_space = cls.obs_space()
        feature_selection = {}
        for entity_name, entity in obs_space.entities.items():
            feature_selection[entity_name] = np.array(
                [entity.features.index(f) for f in entity.features], dtype=np.int32
            )
        return feature_selection

    def env_cls(self) -> Type["Environment"]:
        return self.__class__


class BatchEnv(ABC):
    @abstractmethod
    def reset(self, obs_space: ObsSpace) -> ObsBatch:
        raise NotImplementedError

    @abstractmethod
    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_space: ObsSpace
    ) -> ObsBatch:
        raise NotImplementedError


class VecEnv(ABC):
    @abstractmethod
    def env_cls(cls) -> Type[Environment]:
        """
        Returns the class of the underlying environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, obs_config: ObsSpace) -> List[Observation]:
        raise NotImplementedError

    @abstractmethod
    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_filter: ObsSpace
    ) -> List[Observation]:
        raise NotImplementedError


class BaseEnvList(VecEnv):
    def __init__(
        self, env_cls: Type[Environment], env_kwargs: Dict[str, Any], num_envs: int
    ):
        self.envs = [env_cls(**env_kwargs) for _ in range(num_envs)]  # type: ignore
        self.cls = env_cls

    def env_cls(cls) -> Type[Environment]:
        return cls.cls

    def reset(self, obs_space: ObsSpace) -> List[Observation]:
        return [e.reset(obs_space) for e in self.envs]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_space: ObsSpace
    ) -> List[Observation]:
        observations = []
        for e, a in zip(self.envs, actions):
            obs = e.act(a, obs_space)
            if obs.done:
                # TODO: something is wrong with the interface here
                new_obs = e.reset(obs_space)
                new_obs.done = True
                new_obs.reward = obs.reward
                new_obs.end_of_episode_info = obs.end_of_episode_info
                observations.append(new_obs)
            else:
                observations.append(obs)
        return observations


class EnvList(BatchEnv):
    def __init__(
        self, env_cls: Type[Environment], env_kwargs: Dict[str, Any], num_envs: int
    ):
        self.envs = BaseEnvList(env_cls, env_kwargs, num_envs)

    def reset(self, obs_space: ObsSpace) -> ObsBatch:
        return batch_obs(self.envs.reset(obs_space))

    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_space: ObsSpace
    ) -> ObsBatch:
        observations = self.envs.act(actions, obs_space)
        return batch_obs(observations)


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


def _worker(
    remote: conn.Connection,
    parent_remote: conn.Connection,
    env_list_config: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env_args = env_list_config.var
    envs = BaseEnvList(*env_args)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "act":
                # def act(self, action: Mapping[str, Action], obs_filter: ObsSpace) -> Observation:
                observation = envs.act(data[0], data[1])
                remote.send(observation)
            elif cmd == "reset":
                # def reset(self, obs_filter: ObsSpace) -> Observation:
                observation = envs.reset(data)
                remote.send(observation)
            elif cmd == "close":
                envs.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class ParallelEnvList(BatchEnv):
    """
    We fork the subprocessing from the stable-baselines implementation, but use RaggedBuffers for collecting batches

    Citation here: https://github.com/DLR-RM/stable-baselines3/blob/master/CITATION.bib
    """

    def __init__(
        self,
        env_cls: Type[Environment],
        env_kwargs: Dict[str, Any],
        num_envs: int,
        num_processes: int,
        start_method: Optional[str] = None,
    ):

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        assert (
            num_envs % num_processes == 0
        ), "The required number of environments can not be equally split into the number of specified processes."

        self.num_processes = num_processes
        self.num_envs = num_envs
        self.envs_per_process = int(num_envs / num_processes)

        env_list_configs = [
            (env_cls, env_kwargs, self.envs_per_process)
            for _ in range(self.num_processes)
        ]

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_processes)]
        )
        self.processes = []
        for work_remote, remote, env_list_config in zip(
            self.work_remotes, self.remotes, env_list_configs
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_list_config))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.cls = env_cls

    def env_cls(cls) -> Type[Environment]:
        return cls.cls

    def reset(self, obs_space: ObsSpace) -> ObsBatch:
        for remote in self.remotes:
            remote.send(("reset", obs_space))
        observations = []
        for remote in self.remotes:
            observations.extend(remote.recv())
        return batch_obs(observations)

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()

    def _chunk_actions(
        self, actions: Sequence[Mapping[str, Action]]
    ) -> Generator[Sequence[Mapping[str, Action]], List[Observation], None]:
        for i in range(0, len(actions), self.envs_per_process):
            yield actions[i : i + self.envs_per_process]

    def act(
        self, actions: Sequence[Mapping[str, Action]], obs_space: ObsSpace
    ) -> ObsBatch:
        remote_actions = self._chunk_actions(actions)
        for remote, action in zip(self.remotes, remote_actions):
            remote.send(("act", (action, obs_space)))

        observations = []
        for remote in self.remotes:
            observations.extend(remote.recv())
        return batch_obs(observations)
