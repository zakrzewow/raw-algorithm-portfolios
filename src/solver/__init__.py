import os
from abc import ABC, abstractmethod
from typing import Tuple

from ConfigSpace import Configuration, ConfigurationSpace

from src.instance import Instance


class Solver(ABC):
    CONFIGURATION_SPACE: ConfigurationSpace

    def __init__(self, config: Configuration = None):
        if config is None:
            config = self.CONFIGURATION_SPACE.sample_configuration()
        self.config = config

    @abstractmethod
    def solve(self, instance: Instance) -> Tuple[float, float]:
        pass
