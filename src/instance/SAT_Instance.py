import json
import subprocess
from pathlib import Path

import numpy as np

from src.constant import DATA_DIR, SEED, UBC_SAT_FEATURE_PATH
from src.database import DB
from src.instance.Instance import Instance
from src.instance.InstanceList import InstanceList
from src.log import logger
from src.utils import ResultWithTime, Timer


class SAT_Instance(Instance):
    FEATURES = {
        "nvarsOrig": 0.0,
        "nclausesOrig": 0.0,
        "nvars": 0.0,
        "nclauses": 0.0,
        "reducedVars": 0.0,
        "reducedClauses": 0.0,
        "Pre-featuretime": 0.0,
        "vars-clauses-ratio": 0.0,
        "POSNEG-RATIO-CLAUSE-mean": 0.0,
        "POSNEG-RATIO-CLAUSE-coeff-variation": 0.0,
        "POSNEG-RATIO-CLAUSE-min": 0.0,
        "POSNEG-RATIO-CLAUSE-max": 0.0,
        "POSNEG-RATIO-CLAUSE-entropy": 0.0,
        "VCG-CLAUSE-mean": 0.0,
        "VCG-CLAUSE-coeff-variation": 0.0,
        "VCG-CLAUSE-min": 0.0,
        "VCG-CLAUSE-max": 0.0,
        "VCG-CLAUSE-entropy": 0.0,
        "UNARY": 0.0,
        "BINARY+": 0.0,
        "TRINARY+": 0.0,
        "Basic-featuretime": 0.0,
        "VCG-VAR-mean": 0.0,
        "VCG-VAR-coeff-variation": 0.0,
        "VCG-VAR-min": 0.0,
        "VCG-VAR-max": 0.0,
        "VCG-VAR-entropy": 0.0,
        "POSNEG-RATIO-VAR-mean": 0.0,
        "POSNEG-RATIO-VAR-stdev": 0.0,
        "POSNEG-RATIO-VAR-min": 0.0,
        "POSNEG-RATIO-VAR-max": 0.0,
        "POSNEG-RATIO-VAR-entropy": 0.0,
        "HORNY-VAR-mean": 0.0,
        "HORNY-VAR-coeff-variation": 0.0,
        "HORNY-VAR-min": 0.0,
        "HORNY-VAR-max": 0.0,
        "HORNY-VAR-entropy": 0.0,
        "horn-clauses-fraction": 0.0,
        "VG-mean": 0.0,
        "VG-coeff-variation": 0.0,
        "VG-min": 0.0,
        "VG-max": 0.0,
        "KLB-featuretime": 0.0,
        "CG-mean": 0.0,
        "CG-coeff-variation": 0.0,
        "CG-min": 0.0,
        "CG-max": 0.0,
        "CG-entropy": 0.0,
        "cluster-coeff-mean": 0.0,
        "cluster-coeff-coeff-variation": 0.0,
        "cluster-coeff-min": 0.0,
        "cluster-coeff-max": 0.0,
        "cluster-coeff-entropy": 0.0,
        "CG-featuretime": 0.0,
    }

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

    def _calculate_ubc_features(self) -> dict:
        try:
            logger.debug(f"[{self}] starting...")
            result = subprocess.run(
                [UBC_SAT_FEATURE_PATH, "-base", self.filepath],
                capture_output=True,
                text=True,
            )
            logger.debug(f"[{self}] result={result}")
            output = result.stdout.strip().splitlines()
            # for line in output:
            # logger.debug(f"[{self}] {line}")

            header = output[-2].split(",")
            values = output[-1].split(",")

            feature_dict = {header[i]: float(values[i]) for i in range(len(header))}
            return feature_dict
        except Exception as e:
            logger.error(f"[{self}] error calculating UBC features: {e}")
            return {}

    @classmethod
    def _calculate_features(cls, instance: "Instance") -> ResultWithTime:
        with Timer() as timer:
            ubc_features = instance._calculate_ubc_features()
            features = {**instance.FEATURES, **ubc_features}
        return ResultWithTime(features, timer.elapsed_time)


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
