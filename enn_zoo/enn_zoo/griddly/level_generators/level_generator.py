from abc import ABC, abstractmethod


class LevelGenerator(ABC):
    """
    Abstract Base class for Griddly Level Generators
    """

    @abstractmethod
    def generate(self) -> str:
        """
        Returns a valid Griddly level string
        """
        raise NotImplementedError
