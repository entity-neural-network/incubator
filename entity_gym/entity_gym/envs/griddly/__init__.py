from dataclasses import dataclass


@dataclass
class GriddlyEnvSpec:
    yaml_file: str = ''


GRIDDLY_ENVS: Dict[str, Type[Environment]] = {
    ENNWrapper('env_descriptions/clusters.yaml')
}
