import json
from pathlib import Path
from typing import Tuple

import numpy as np

from src.constant import DATA_DIR
from src.instance import Instance, InstanceSet


class TSP_Instance(Instance):
    def __init__(self, filepath: Path, optimum: float):
        self.filepath = filepath
        self.optimum = optimum

    def __hash__(self):
        path_tuple = self.filepath.parts
        data_idx = path_tuple.index("data")
        return "/".join(path_tuple[data_idx:])


class TSP_InstanceSet(InstanceSet):
    def __init__(self):
        super().__init__()

    @classmethod
    def train_test_from_index_file(
        cls,
        filepath: Path,
        train_size: int,
        seed: int = 0,
    ) -> Tuple["TSP_InstanceSet", "TSP_InstanceSet"]:
        train_instances = cls()
        test_instances = cls()

        with open(filepath) as f:
            index = json.load(f)

        instances = []
        for k, v in index.items():
            filepath = DATA_DIR / Path(k)
            instance = TSP_Instance(filepath, v)
            instances.append(instance)

        np.random.seed(seed)
        np.random.shuffle(instances)
        train_instances.extend(instances[:train_size])
        test_instances.extend(instances[train_size:])
        return train_instances, test_instances
