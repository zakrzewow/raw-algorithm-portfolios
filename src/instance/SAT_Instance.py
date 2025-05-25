import json
from pathlib import Path

import numpy as np

from src.constant import DATA_DIR, SEED
from src.database import DB
from src.instance.Instance import Instance
from src.instance.InstanceList import InstanceList
from src.utils import ResultWithTime


class SAT_Instance(Instance):
    FEATURES = {}

    def __init__(
        self,
        filepath: Path,
        max_cost: float = 0,
        max_time: float = 0,
    ):
        super().__init__()
        self.filepath = filepath
        self.max_cost = max_cost
        self.max_time = max_time

    def __repr__(self):
        filepath = self._get_short_filepath()
        str_ = f"SAT_Instance(filepath={filepath})"
        return str_

    def to_dict(self) -> dict:
        return {
            "filepath": self._get_short_filepath(),
            "max_cost": self.max_cost,
            "max_time": self.max_time,
            **self.features,
        }

    def _get_short_filepath(self) -> str:
        path_parts = self.filepath.parts
        data_dir_parts = DATA_DIR.parts
        filepath = "/".join(path_parts[len(data_dir_parts) :])
        return filepath

    @classmethod
    def from_db(cls, id_: str) -> "SAT_Instance":
        dict_ = DB().select_id(DB.SCHEMA.INSTANCES, id_)
        filepath = DATA_DIR / dict_["filepath"]
        optimum = dict_["optimum"]
        max_cost = dict_["max_cost"]
        max_time = dict_["max_time"]
        instance = cls(filepath, optimum, max_cost, max_time)
        del dict_["filepath"]
        del dict_["optimum"]
        instance.features = dict_
        return instance

    @classmethod
    def _calculate_features(cls, instance: "Instance") -> ResultWithTime:
        return ResultWithTime({}, 0.0)


def SAT_from_index_file(
    filepath: Path,
    max_cost: float = 0.0,
    max_time: float = 0.0,
) -> InstanceList:
    instances = InstanceList()

    with open(filepath) as f:
        index = json.load(f)

    for v in index:
        filepath = DATA_DIR / Path(v)
        instance = SAT_Instance(filepath, max_cost, max_time)
        instances.append(instance)
    return instances


def TSP_train_test_from_index_file(
    filepath: Path,
    train_size: int,
    max_cost: float,
    max_time: float,
) -> tuple[InstanceList, InstanceList]:
    train_instances = InstanceList()
    test_instances = InstanceList()
    instances = SAT_from_index_file(filepath, max_cost, max_time)

    rng = np.random.default_rng(SEED)
    rng.shuffle(instances)
    train_instances.extend(instances[:train_size])
    test_instances.extend(instances[train_size:])
    return train_instances, test_instances
