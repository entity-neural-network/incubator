import functools
import os
from typing import Any, Callable, Dict, Optional, Type

from griddly import gd

from enn_zoo.griddly_env.level_generators.clusters_generator import (
    ClustersLevelGenerator,
)
from enn_zoo.griddly_env.level_generators.crafter_generator import CrafterLevelGenerator
from enn_zoo.griddly_env.wrappers.grafter_env import grafter_env
from enn_zoo.griddly_env.wrappers.griddly_env import GriddlyEnv

init_path = os.path.dirname(os.path.realpath(__file__))


def create_env(
    env_wrapper: Optional[Callable[[Dict[str, Any]], Type[GriddlyEnv]]] = None,
    **kwargs: Any
) -> Type[GriddlyEnv]:
    def partialclass() -> Type[GriddlyEnv]:
        """
        Make the __init__ partial so we can override the kwargs we are putting here with custom ones later
        """

        default_args: Dict[str, Any] = {}

        default_args["player_observer_type"] = gd.ObserverType.ENTITY
        default_args["global_observer_type"] = gd.ObserverType.SPRITE_2D

        default_args.update(kwargs)

        if env_wrapper is not None:
            return env_wrapper(**default_args)  # type: ignore
        else:

            class InstantiatedGriddlyEnv(GriddlyEnv):
                __init__ = functools.partialmethod(GriddlyEnv.__init__, **default_args)  # type: ignore

            return InstantiatedGriddlyEnv  # type: ignore

    return partialclass()


GRIDDLY_ENVS: Dict[str, Dict[str, Any]] = {
    "GDY-Clusters-Multi-Generated-Small": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 10,
                "height": 10,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 5,
                "m_blue": 5,
                "m_green": 5,
                "m_spike": 5,
                "walls": False,
                "avatar": False,
            }
        ),
    },
    "GDY-Clusters-Multi-Generated-Medium": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 20,
                "height": 20,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 10,
                "m_blue": 10,
                "m_green": 10,
                "m_spike": 10,
                "walls": False,
                "avatar": False,
            }
        ),
    },
    "GDY-Clusters-Multi-Generated-Large": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 50,
                "height": 50,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 20,
                "m_blue": 20,
                "m_green": 20,
                "m_spike": 20,
                "walls": False,
                "avatar": False,
            }
        ),
    },
    "GDY-Clusters-Multi-All": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "random_levels": True,
    },
    "GDY-Clusters-Multi-0": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 0,
    },
    "GDY-Clusters-Multi-1": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 1,
    },
    "GDY-Clusters-Multi-2": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 2,
    },
    "GDY-Clusters-Multi-3": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 3,
    },
    "GDY-Clusters-Multi-4": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters_entities.yaml"),
        "level": 4,
    },
    "GDY-Clusters-Generated-Small": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 10,
                "height": 10,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 5,
                "m_blue": 5,
                "m_green": 5,
                "m_spike": 5,
                "walls": False,
                "avatar": True,
            }
        ),
    },
    "GDY-Clusters-Generated-Medium": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 20,
                "height": 20,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 10,
                "m_blue": 10,
                "m_green": 10,
                "m_spike": 10,
                "walls": False,
                "avatar": True,
            }
        ),
    },
    "GDY-Clusters-Generated-Large": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level_generator": ClustersLevelGenerator(
            {
                "width": 50,
                "height": 50,
                "p_red": 1.0,
                "p_green": 1.0,
                "p_blue": 1.0,
                "m_red": 20,
                "m_blue": 20,
                "m_green": 20,
                "m_spike": 20,
                "walls": False,
                "avatar": True,
            }
        ),
    },
    "GDY-Clusters-All": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "random_levels": True,
    },
    "GDY-Clusters-0": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 0,
    },
    "GDY-Clusters-1": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 1,
    },
    "GDY-Clusters-2": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 2,
    },
    "GDY-Clusters-3": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 3,
    },
    "GDY-Clusters-4": {
        "yaml_file": os.path.join(init_path, "env_descriptions/clusters.yaml"),
        "level": 4,
    },
    ############ Grafter Envs ############
    "GDY-Grafter-Single-30": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_single.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 1),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-Single-50": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_single.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 50, 50, 1),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-Single-100": {
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_single.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 100, 100, 1),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-4Player-30": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_multi_4.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 4),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-4Player-50": {
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/rafter_multi_4.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 50, 50, 4),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-4Player-100": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_multi_4.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 100, 100, 4),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-8Player-50": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_multi_8.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 30, 30, 8),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-8Player-100": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_multi_8.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 100, 100, 8),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
    "GDY-Grafter-8Player-200": {
        "env_wrapper": grafter_env,
        "yaml_file": os.path.join(
            init_path, "env_descriptions/grafter/grafter_multi_8.yaml"
        ),
        "level_generator": CrafterLevelGenerator(100, 200, 200, 8),
        "image_path": os.path.join(init_path, "images/grafter"),
    },
}
