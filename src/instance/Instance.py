import concurrent.futures
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
        self.cut_off_cost = 0.0
        self.cut_off_time = 0.0

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

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    def calculate_features(
        self,
        executor: concurrent.futures.ProcessPoolExecutor = None,
    ) -> ResultWithTime:
        if executor is None:
            result_with_time = self._calculate_features(self)
            self.features = result_with_time.result
            return result_with_time
        else:
            future = executor.submit(self._calculate_features, self)
            try:
                result_with_time = future.result(300)
                self.features = result_with_time.result
                return result_with_time
            except Exception as e:
                logger.error(f"Error calculating features: {e}")
                self.features = {}
                return ResultWithTime(None, 0.0)

    @classmethod
    @abstractmethod
    def _calculate_features(cls, instance: "Instance") -> ResultWithTime:
        pass

    @property
    def features_calculated(self):
        return len(self.features) > 0

    def get_array(self) -> np.array:
        return np.array(list(self.features.values()))
