from abc import ABC
from typing import List


class Instance(ABC):
    pass


class InstanceSet(ABC):
    def __init__(self):
        self._set: List[Instance] = []

    def __len__(self):
        return len(self._set)

    def __getitem__(self, item):
        return self._set[item]

    def __iter__(self):
        return iter(self._set)

    def append(self, instance: Instance):
        self._set.append(instance)

    def extend(self, instances: List[Instance]):
        self._set.extend(instances)
