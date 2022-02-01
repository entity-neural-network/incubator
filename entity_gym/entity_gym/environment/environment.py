from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import numpy as np
import numpy.typing as npt


EntityID = Any
EntityType = str
ActionType = str


@dataclass
class CategoricalActionSpace:
    choices: List[str]


@dataclass
class SelectEntityActionSpace:
    pass


ActionSpace = Union[CategoricalActionSpace, SelectEntityActionSpace]


@dataclass
class CategoricalActionMask:
    """
    Action mask for categorical action that specifies which agents can perform the action,
    and includes a dense mask that further contraints the choices available to each agent.
    """

    actor_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actor_types: Optional[Sequence[EntityType]] = None
    """
    The types of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    mask: Optional[np.ndarray] = None
    """
    A boolean array of shape (len(actors), len(choices)). If mask[i, j] is True, then
    agent i can perform action j.
    """

    def __post_init__(self) -> None:
        assert (
            self.actor_ids is None or self.actor_types is None
        ), "Only one of actor_ids or actor_types can be specified"


@dataclass
class SelectEntityActionMask:
    """
    Action mask for select entity action that specifies which agents can perform the action,
    and includes a dense mask that further contraints what other entities can be selected by
    each actor.
    """

    actor_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actor_types: Optional[Sequence[EntityType]] = None
    """
    The types of the entities that can perform the action.
    If None, all entities can perform the action.
    """

    actee_types: Optional[Sequence[EntityType]] = None
    """
    The types of entities that can be selected by each actor.
    If None, all entities types can be selected by each actor.
    """

    actee_ids: Optional[Sequence[EntityID]] = None
    """
    The ids of the entities of each type that can be selected by each actor.
    If None, all entities can be selected by each actor.
    """

    mask: Optional[npt.NDArray[np.bool_]] = None
    """
    An boolean array of shape (len(actors), len(actees)). If mask[i, j] is True, then
    agent i can select entity j.
    (NOT CURRENTLY IMPLEMENTED)
    """

    def __post_init__(self) -> None:
        assert (
            self.actor_ids is None or self.actor_types is None
        ), "Only one of actor_ids or actor_types can be specified"
        assert (
            self.actee_types is None or self.actee_ids is None
        ), "Either actee_entity_types or actees can be specified, but not both."


ActionMask = Union[CategoricalActionMask, SelectEntityActionMask]


@dataclass
class EpisodeStats:
    length: int
    total_reward: float


@dataclass
class Entity:
    features: List[str]


@dataclass
class ObsSpace:
    entities: Dict[str, Entity]


@dataclass
class Observation:
    features: Mapping[EntityType, npt.NDArray[np.float32]]
    actions: Mapping[ActionType, ActionMask]
    done: bool
    reward: float
    ids: Mapping[EntityType, Sequence[EntityID]] = field(default_factory=dict)

    end_of_episode_info: Optional[EpisodeStats] = None

    def __post_init__(self) -> None:
        self._id_to_index: Optional[Dict[EntityID, int]] = None
        self._index_to_id: Optional[List[EntityID]] = None

    def _actor_indices(
        self, atype: ActionType, obs_space: ObsSpace
    ) -> npt.NDArray[np.int64]:
        action = self.actions[atype]
        if action.actor_ids is not None:
            id_to_index = self.id_to_index(obs_space)
            return np.array(
                [id_to_index[id] for id in action.actor_ids], dtype=np.int64
            )
        elif action.actor_types is not None:
            ids: List[int] = []
            id_to_index = self.id_to_index(obs_space)
            for etype in action.actor_types:
                ids.extend(id_to_index[id] for id in self.ids[etype])
            return np.array(ids, dtype=np.int64)
        else:
            return np.arange(  # type: ignore
                sum(len(self.ids[etype]) for etype in obs_space.entities),
                dtype=np.int64,
            )

    def _actee_indices(
        self, atype: ActionType, obs_space: ObsSpace
    ) -> npt.NDArray[np.int64]:
        action = self.actions[atype]
        assert isinstance(action, SelectEntityActionMask)
        if action.actee_ids is not None:
            id_to_index = self.id_to_index(obs_space)
            return np.array(
                [id_to_index[id] for id in action.actee_ids], dtype=np.int64
            )
        else:
            return np.arange(  # type: ignore
                sum(len(self.ids[etype]) for etype in obs_space.entities),
                dtype=np.int64,
            )

    def id_to_index(self, obs_space: ObsSpace) -> Dict[EntityID, int]:
        offset = 0
        if self._id_to_index is None:
            self._id_to_index = {}
            for etype in obs_space.entities.keys():
                ids = self.ids.get(etype)
                if ids is None:
                    continue
                for i, id in enumerate(ids):
                    self._id_to_index[id] = i + offset
                offset += len(ids)
        return self._id_to_index

    def index_to_id(self, obs_space: ObsSpace) -> List[EntityID]:
        if self._index_to_id is None:
            self._index_to_id = []
            for etype in obs_space.entities.keys():
                ids = self.ids.get(etype)
                if ids is None:
                    ids = [None] * self.features[etype].shape[0]
                self._index_to_id.extend(ids)
        return self._index_to_id


@dataclass
class CategoricalAction:
    actors: Sequence[EntityID]
    actions: npt.NDArray[np.int64]

    def items(self) -> Generator[Tuple[EntityID, int], None, None]:
        for i, j in zip(self.actors, self.actions):
            yield i, j


@dataclass
class SelectEntityAction:
    actors: Sequence[EntityID]
    actees: Sequence[EntityID]

    def items(self) -> Generator[Tuple[EntityID, EntityID], None, None]:
        for i, j in zip(self.actors, self.actees):
            yield i, j


Action = Union[CategoricalAction, SelectEntityAction]


class Environment(ABC):
    """
    Abstraction over reinforcement learning environments with observations based on structured lists of entities.
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
    def reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, action: Mapping[ActionType, Action]) -> Observation:
        """
        Performs the given action and returns the resulting observation.

        Args:
            action: Maps the name of each action type to the action to perform.
        """
        raise NotImplementedError

    def reset_filter(self, obs_filter: ObsSpace) -> Observation:
        return self.__class__.filter_obs(self.reset(), obs_filter)

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        """
        Renders the environment

        Args:
            **kwargs: a dictionary of arguments to send to the rendering process
        """
        raise NotImplementedError

    def act_filter(
        self, action: Mapping[ActionType, Action], obs_filter: ObsSpace
    ) -> Observation:
        return self.__class__.filter_obs(self.act(action), obs_filter)

    def close(self) -> None:
        pass

    @classmethod
    def filter_obs(cls, obs: Observation, obs_filter: ObsSpace) -> Observation:
        selectors = cls._compile_feature_filter(obs_filter)
        return Observation(
            features={
                etype: features[:, selectors[etype]].reshape(
                    features.shape[0], len(selectors[etype])
                )
                for etype, features in obs.features.items()
            },
            actions=obs.actions,
            done=obs.done,
            reward=obs.reward,
            ids=obs.ids,
            end_of_episode_info=obs.end_of_episode_info,
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
