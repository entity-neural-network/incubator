from dataclasses import dataclass
from entity_gym.envs.griddly.enn_wrapper import ENNWrapper
from typing import Type, Dict

from entity_gym.environment import Environment


def create_clusters():
    return ENNWrapper('entity_gym/entity_gym/envs/griddly/env_descriptions/clusters.yaml')


GRIDDLY_ENVS: Dict[str, Type[Environment]] = {
    "GDY-Clusters": create_clusters
}
