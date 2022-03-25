from os import listdir
from pathlib import Path

import hyperstate

from enn_ppo.train import TrainConfig


def test_configs() -> None:
    config_dir = Path(__file__).parent.parent.parent.parent / "configs/entity-gym"
    for config_file in listdir(config_dir):
        hyperstate.load(TrainConfig, config_dir / config_file)
