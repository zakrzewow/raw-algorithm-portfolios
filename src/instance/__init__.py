from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.log import logger


class Instance(ABC):
    FEATURES = {}

    def __eq__(self, value):
        return hash(self) == hash(value)

    @abstractmethod
    def calculate_features(self) -> Tuple[float, Dict]:
        pass

    def mutate(self) -> Tuple["Instance", float]:
        pass

    def log(self):
        logger.debug(str(self.__hash__()))


class InstanceSet(ABC):
    def __init__(self):
        self._set: List[Instance] = []

    @classmethod
    def from_instance_list(cls, instances: List[Instance]) -> "InstanceSet":
        instance_set = cls()
        instance_set.extend(instances)
        return instance_set

    def __len__(self):
        return len(self._set)

    @property
    def size(self):
        return len(self)

    def __getitem__(self, item):
        return self._set[item]

    def __iter__(self):
        return iter(self._set)

    def append(self, instance: Instance):
        self._set.append(instance)

    def extend(self, instances: List[Instance]):
        self._set.extend(instances)

    def copy(self) -> "InstanceSet":
        return self.from_instance_list(self._set)

    def log(self):
        logger.debug(f"  InstanceSet[size={self.size}]  ".center(80, "="))
        for instance in self._set:
            instance.log()
        logger.debug("=" * 80)
