import base64
import copy
import hashlib
from abc import ABC, abstractmethod
from typing import Tuple

from ConfigSpace import Configuration, ConfigurationSpace

from src.instance import Instance


class Solver(ABC):
    CONFIGURATION_SPACE: ConfigurationSpace
    MAX_COST = 0.0

    def __init__(self, config: Configuration = None):
        if config is None:
            config = self.CONFIGURATION_SPACE.sample_configuration()
        self.config = config

    def __eq__(self, value):
        return hash(self) == hash(value)

    @abstractmethod
    def solve(self, instance: Instance) -> Tuple[float, float]:
        pass

    def copy(self) -> "Solver":
        return copy.deepcopy(self)

    def __hash__(self):
        str_ = ";".join([f"{k}={v}" for k, v in self.config.items()])
        sha256_hash = hashlib.sha256(str_.encode()).digest()
        base64_hash = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")
        return base64_hash
