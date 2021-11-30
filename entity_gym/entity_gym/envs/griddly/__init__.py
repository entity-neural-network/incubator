from dataclasses import dataclass
from entity_gym.envs.griddly.enn_wrapper import ENNWrapper
from typing import Type, Dict
import os

from entity_gym.environment import Environment

init_path = os.path.dirname(os.path.realpath(__file__))


def create_clusters():
    return ENNWrapper(os.path.join(init_path, "env_descriptions/clusters.yaml"))


GRIDDLY_ENVS: Dict[str, Type[Environment]] = {"GDY-Clusters": create_clusters}
