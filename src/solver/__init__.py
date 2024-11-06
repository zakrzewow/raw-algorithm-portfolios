import os
from abc import ABC, abstractmethod
from typing import Tuple

from ConfigSpace import Configuration

from src.instance import Instance


class Solver(ABC):
    def __init__(self, config: Configuration):
        self.config = config

    @abstractmethod
    def solve(self, instance: Instance) -> Tuple[float, float]:
        pass

    @property
    def _pid(self) -> int:
        return os.getpid()
