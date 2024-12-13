import base64
import copy
import hashlib
from abc import ABC, abstractmethod
from typing import Tuple

from ConfigSpace import Configuration, ConfigurationSpace

from src.instance import Instance
from src.log import logger


class Solver(ABC):
    CONFIGURATION_SPACE: ConfigurationSpace
    MAX_COST = 0.0
    MAX_TIME = 0.0

    def __init__(self, config: Configuration = None):
        if config is None:
            config = self.CONFIGURATION_SPACE.sample_configuration()
        self.config = config

    @property
    def max_cost_time(self) -> Tuple[float, float]:
        return self.MAX_COST, self.MAX_TIME

    @abstractmethod
    def solve(self, instance: Instance) -> Tuple[float, float]:
        pass

    def copy(self) -> "Solver":
        return copy.deepcopy(self)

    def __hash__(self):
        str_ = ";".join([f"{k}={v}" for k, v in self.config.items()])
        sha256_hash = hashlib.sha256(str_.encode()).digest()
        hash_int = int.from_bytes(sha256_hash, byteorder="big")
        return hash_int

    def __eq__(self, other):
        return hash(self) == hash(other)

    def log(self):
        str_ = f"{hash(self)} --> "
        str_ += ";".join([f"{k}={v}" for k, v in self.config.items()])
        logger.debug(str_)
        return str_
