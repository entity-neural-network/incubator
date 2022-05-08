"""
The envionment module defines the core interfaces that make up an `Environment <#entity_gym.environment.Environment>`_.

Actions
-------

Actions are how agents interact with the environment.
There are three parts to every action:

* :code:`ActionSpace` defines the shape of the action. For example, a categorical action space consisting of the 4 discrete choices "up", "down", "left", and "right".
* :code:`ActionMask` is used to further constrain the available actions on a specific timestep. For example, only "up" and "down" may be available some timestep.
* :code:`Action` represent the actual action that is chosen by an agent. For example, the "down" action may have been chosen.

There are curently three different action spaces:

* `GlobalCategoricalActionSpace <#entity_gym.environment.GlobalCategoricalActionSpace>`_ allows the agent to choose a single option from a discrete set of actions.
* `CategoricalActionSpace <#entity_gym.environment.CategoricalActionSpace>`_ allows multiple entities to choose a single option from a discrete set of actions.
* `SelectEntityActionSpace <#entity_gym.environment.SelectEntityActionSpace>`_ allows multiple entities to choose another entity.

Observations
------------

Observations are how agents receive information from the environment.
Each `Environment <#entity_gym.environment.Environment>`_ must define an `ObsSpace <#entity_gym.environment.ObsSpace>`_, which specifies the shape of the observations returned by this environment.
On each timestep, the environment returns an `Observation <#entity_gym.environment.Observation>`_ object, which contains all the entities and features that are visible to the agent.
"""
from .env_list import *
from .environment import *
from .parallel_env_list import *
from .vec_env import *
from .action import *

__all__ = [
    "Environment",
    # Observation
    "ObsSpace",
    "Entity",
    "Observation",
    "EntityName",
    "ActionName",
    "EntityID",
    # Action
    "Action",
    "ActionSpace",
    "ActionMask",
    "CategoricalActionSpace",
    "CategoricalAction",
    "CategoricalActionMask",
    "GlobalCategoricalActionSpace",
    "GlobalCategoricalAction",
    "GlobalCategoricalActionMask",
    "SelectEntityAction",
    "SelectEntityActionSpace",
    "SelectEntityActionMask",
    # VecEnv
    "VecEnv",
    "EnvList",
    "ParallelEnvList",
    "VecActionMask",
    "VecObs",
    "VecCategoricalActionMask",
    "VecSelectEntityActionMask",
]
