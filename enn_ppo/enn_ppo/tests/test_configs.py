from os import listdir
from enn_ppo.train import TrainConfig
import hyperstate
from pathlib import Path


def test_configs() -> None:
    config_dir = Path(__file__).parent.parent.parent.parent / "configs/entity-gym"
    for config_file in listdir(config_dir):
        hyperstate.load(TrainConfig, config_dir / config_file)
