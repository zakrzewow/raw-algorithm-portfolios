from abc import ABC, abstractmethod
from typing import Dict, List


class Instance(ABC):
    FEATURES = {}

    def __init__(self):
        self.features = None

    def __eq__(self, value):
        return hash(self) == hash(value)

    @abstractmethod
    def calculate_features(self) -> Dict:
        pass

    def set_features(self, features: Dict):
        self.features = features


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
