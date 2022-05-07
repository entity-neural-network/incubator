from hyperstate import StateManager

from .config import TrainConfig
from .train import State, initialize


def load_checkpoint(path: str) -> StateManager[TrainConfig, State]:
    return StateManager(TrainConfig, State, initialize, init_path=path)


def load_agent(path: str) -> State:
    return StateManager(TrainConfig, State, initialize, init_path=path).state
