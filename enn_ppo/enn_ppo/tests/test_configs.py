from os import listdir
from pathlib import Path

import hyperstate
from hyperstate.schema.schema_change import Severity
from hyperstate.schema.schema_checker import SchemaChecker
from hyperstate.schema.types import load_schema

from enn_ppo.config import TrainConfig


def test_schema() -> None:
    schema_files = Path(__file__).parent.parent.parent / "config-schema.ron"
    path = str(schema_files)
    print(path)
    with open(path) as f:
        f.read()
    old = load_schema(path)
    checker = SchemaChecker(old, TrainConfig)
    if checker.severity() >= Severity.WARN:
        checker.print_report()
    assert checker.severity() == Severity.INFO


def test_configs() -> None:
    for subdir in ["entity-gym", "procgen"]:
        config_dir = Path(__file__).parent.parent.parent.parent / "configs" / subdir
        for config_file in listdir(config_dir):
            hyperstate.load(TrainConfig, config_dir / config_file)
