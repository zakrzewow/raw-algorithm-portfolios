import copy
from typing import Iterable

from src.instance.Instance import Instance
from src.log import logger


class InstanceList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_iterable(cls, instances: Iterable[Instance]) -> "InstanceList":
        instance_list = cls()
        instance_list.extend(instances)
        return instance_list

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return InstanceList(result)
        return result

    def __repr__(self):
        str_ = super().__repr__()
        str_ = f"InstanceList(size={self.size}){str_}"
        return str_

    def copy(self) -> "InstanceList":
        return copy.deepcopy(self)

    def log(self):
        logger.debug(self.__repr__())

    @property
    def size(self):
        return len(self)
