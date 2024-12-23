import copy
from abc import ABC, abstractmethod

import numpy as np

from src.constant import SEED
from src.database import DB
from src.log import logger
from src.utils import ResultWithTime, hash_str


class Instance(ABC):
    FEATURES = {}

    def __init__(self):
        self.features = {}
        self._rng = np.random.default_rng(SEED)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        str_ = self.__repr__()
        return hash_str(str_)

    @abstractmethod
    def __repr__(self):
        pass

    def copy(self) -> "Instance":
        return copy.deepcopy(self)

    def id(self):
        return str(hash(self))

    def log(self):
        logger.debug(self.__repr__())

    @classmethod
    @abstractmethod
    def from_db(cls, id_: str) -> "Instance":
        pass

    def to_db(self):
        DB().insert(DB.SCHEMA.INSTANCES, self.id(), self.to_dict())
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def calculate_features(self) -> ResultWithTime:
        pass

    @abstractmethod
    def mutate(self) -> ResultWithTime:
        pass

    @property
    def features_calculated(self):
        return len(self.features) > 0
