from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import random


@dataclass
class ActionSpace:
    name: str


@dataclass
class Categorical(ActionSpace):
    n: int
    choice_labels: Optional[List[str]] = None


@dataclass
class SelectEntity(ActionSpace):
    pass


@dataclass
class ObsConfig:
    entities: Dict[str, List[str]]


@dataclass
class AvailableActions(ABC):
    # Indices of entities which can perform the action on this time step.
    actors: List[int]
    # Action mask with dimensions (len(actors), n_choices) if this is a categorical action,
    # or (len(actors), len(entities)) if this is a select object action.
    mask: Optional[np.ndarray]
    # TODO: also support sparse action masks for select object actions that specify the actees.


@dataclass
class Observation:
    # Maps each entity type to an array with features for each entity of that type.
    entities: Sequence[Tuple[str, np.ndarray]]
    # Maps each entity index to an identifier for the entity.
    ids: Sequence[int]
    # Maps each action type to a list of indices of the entities that can perform that action.
    actions: Sequence[Tuple[str, Sequence[int]]]
    reward: float
    done: bool


@dataclass
class Entity:
    name: str
    features: List[str]


@dataclass
class Action:
    # Maps each action type to a list of tuples.
    # The first element of each tuple is the id of the entity that performs the action.
    # If this is a categorical action, the second element is the index of the action.
    # If this is a select object action, the second element is the id of the selected object.
    chosen_actions: Dict[str, Sequence[Tuple[int, int]]]


class Environment(ABC):
    @classmethod
    def entities(cls) -> List[Entity]:
        raise NotImplementedError

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        raise NotImplementedError

    @classmethod
    def entity_dict(cls) -> Dict[str, Entity]:
        return {e.name: e for e in cls.entit915ies()}

    @classmethod
    def action_space_dict(cls) -> Dict[str, ActionSpace]:
        return {a.name: a for a in cls.action_space()}

    def reset(self, obs_config: ObsConfig) -> Observation:
        raise NotImplementedError

    def act(self, action: Action, obs_config: ObsConfig) -> Observation:
        raise NotImplementedError


@dataclass
class MoveToOrigin(Environment):
    x_pos: float = 0.0
    y_pos: float = 0.0
    x_velocity: float = 0.0
    y_velocity: float = 0.0
    last_x_pos = 0.0
    last_y_pos = 0.0
    step: int = 0

    @classmethod
    def entities(cls) -> List[Entity]:
        return [
            Entity(
                name="Spaceship",
                features=["x_pos", "y_pos", "x_velocity", "y_velocity", "step"]
            ),
        ]

    @classmethod
    def action_space(cls) -> List[ActionSpace]:
        return [
            Categorical(
                name="horizontal_thruster",
                n=5,
                choice_labels=["100% right", "10% right",
                               "hold", "10% left", "100% left"],
            ),
            Categorical(
                name="vertical_thruster",
                n=5,
                choice_labels=["100% up", "10% up",
                               "hold", "10% down", "100% down"],
            ),
        ]

    def reset(self, obs_config: ObsConfig) -> Observation:
        angle = random.uniform(0, 2 * np.pi)
        self.x_pos = np.cos(angle)
        self.y_pos = np.sin(angle)
        self.last_x_pos = self.x_pos
        self.last_y_pos = self.y_pos
        self.x_velocity = 0
        self.y_velocity = 0
        return self.observe(obs_config)

    def act(self, action: Action, obs_config: ObsConfig) -> Observation:
        self.step += 1

        for action_name, chosen_actions in action.chosen_actions.items():
            if action_name == "horizontal_thruster":
                for actor_id, choice_id in chosen_actions:
                    if choice_id == 0:
                        self.x_velocity += 0.01
                    elif choice_id == 1:
                        self.x_velocity += 0.001
                    elif choice_id == 2:
                        pass
                    elif choice_id == 3:
                        self.x_velocity -= 0.001
                    elif choice_id == 4:
                        self.x_velocity -= 0.01
                    else:
                        raise ValueError(f"Invalid choice id {choice_id}")
            elif action_name == "vertical_thruster":
                for actor_id, choice_id in chosen_actions:
                    if choice_id == 0:
                        self.y_velocity += 0.01
                    elif choice_id == 1:
                        self.y_velocity += 0.001
                    elif choice_id == 2:
                        pass
                    elif choice_id == 3:
                        self.y_velocity -= 0.001
                    elif choice_id == 4:
                        self.y_velocity -= 0.01
                    else:
                        raise ValueError(f"Invalid choice id {choice_id}")
            else:
                raise ValueError(f"Unknown action type {action_name}")

        self.last_x_pos = self.x_pos
        self.last_y_pos = self.y_pos

        self.x_pos += self.x_velocity
        self.y_pos += self.y_velocity

        done = self.step >= 32
        return self.observe(obs_config, done)

    def observe(self, obs_config: ObsConfig, done: bool = False) -> Observation:
        entities = []
        for entity_name, features in obs_config.entities.items():
            if entity_name == "Spaceship":
                feature_vals = []
                for feature in features:
                    if feature == "x_pos":
                        feature_vals.append(self.x_pos)
                    elif feature == "y_pos":
                        feature_vals.append(self.y_pos)
                    elif feature == "x_velocity":
                        feature_vals.append(self.x_velocity)
                    elif feature == "y_velocity":
                        feature_vals.append(self.y_velocity)
                    elif feature == "step":
                        feature_vals.append(self.step)
                    else:
                        raise ValueError(f"Unknown feature {feature}")
                entities.append(
                    (entity_name, np.array(feature_vals).reshape(1, -1)))

        return Observation(
            entities=entities,
            actions=[("horizontal_thruster", [0]), ("vertical_thruster", [0])],
            ids=[0],
            reward=(self.last_x_pos ** 2 + self.last_y_pos ** 2) ** 0.5 -
            (self.x_pos ** 2 + self.y_pos ** 2) ** 0.5,
            done=done,
        )


if __name__ == "__main__":
    env = MoveToOrigin()
    obs_config = ObsConfig(
        entities={
            entity.name: entity.features for entity in MoveToOrigin.entities()},
    )

    total_reward = 0
    actions = MoveToOrigin.action_space_dict()
    obs = env.reset(obs_config)
    while not obs.done:
        total_reward += obs.reward
        print(f"Reward: {obs.reward}")
        print(f"Total reward: {total_reward}")
        entity_index = 0
        for entity_type, features in obs.entities:
            for entity in range(features.shape[0]):
                print(
                    f"{obs.ids[entity_index]}: {entity_type}({', '.join(map(lambda nv: nv[0] + '=' + str(nv[1]), zip(obs_config.entities[entity_type], features[entity, :])))})")
                entity_index += 1

        chosen_actions = defaultdict(list)
        for action_name, actors in obs.actions:
            action_def = actions[action_name]
            for actor_id in actors:
                if isinstance(action_def, Categorical):
                    # Prompt user for action
                    print(f"Choose {action_name}")
                    for i in range(action_def.n):
                        print(f"{i}: {action_def.choice_labels[i]}")
                    choice_id = int(input())
                    chosen_actions[action_name].append((actor_id, choice_id))
                else:
                    raise ValueError(f"Unknown action type {action_def}")
        obs = env.act(Action(chosen_actions), obs_config)
