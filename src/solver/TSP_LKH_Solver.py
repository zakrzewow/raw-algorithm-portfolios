import os
import subprocess
from pathlib import Path
from typing import Tuple

from ConfigSpace import Configuration

from src.configuration_space.LKH import CONFIGURATION_SPACE
from src.constant import LKH_PATH, TEMP_DIR
from src.instance import TSP_Instance
from src.solver import Solver


class TSP_LKH_Solver(Solver):
    CONFIGURATION_SPACE = CONFIGURATION_SPACE
    MAX_COST = 100.0
    MAX_TIME = 10.0

    def __init__(self, config: Configuration = None):
        super().__init__(config)

    def solve(self, instance: TSP_Instance) -> Tuple[float, float]:
        config_filepath = self._to_config_file(instance.filepath, instance.optimum)
        result = subprocess.run(
            [LKH_PATH, config_filepath],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        time = self._parse_result(result)
        cost = time if time < self.MAX_TIME else time * 10
        self._remove_config_file(config_filepath)
        return cost, time

    def _to_config_file(self, instance_filepath: Path, optimum: float) -> Path:
        config_filepath = TEMP_DIR / f"config_{os.getpid()}.par"
        with open(config_filepath, "w") as f:
            f.write(f"PROBLEM_FILE = {instance_filepath}\n")
            f.write(f"OPTIMUM = {optimum}\n")
            f.write(f"TRACE_LEVEL = 0\n")
            f.write(f"TOTAL_TIME_LIMIT = {self.MAX_TIME}\n")
            f.write(f"STOP_AT_OPTIMUM = YES\n")
            f.write(f"RUNS = 10000\n")
            for k, v in self.config.items():
                f.write(f"{k} = {v}\n")
        return config_filepath

    def _parse_result(self, result: subprocess.CompletedProcess) -> Tuple[float, float]:
        time = None
        for line in result.stdout.splitlines():
            if "Time.total" in line:
                time = float(line.split()[-2])
                break
        if time is None:
            raise Exception("Time.total not found")
        return min(time, self.MAX_TIME)

    def _remove_config_file(self, config_filepath: Path):
        config_filepath.unlink()
