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
    TOTAL_TIME_LIMIT = 10.0
    CONFIGURATION_SPACE = CONFIGURATION_SPACE

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
        cost = time if time < self.TOTAL_TIME_LIMIT else time * 10
        self.remove_config_file(config_filepath)
        return cost, time

    def _to_config_file(self, problem_filepath: str, optimum: float) -> Path:
        config_filepath = TEMP_DIR / f"config_{os.getpid()}.par"
        with open(config_filepath, "w") as f:
            f.write(f"PROBLEM_FILE = {problem_filepath}\n")
            f.write(f"OPTIMUM = {optimum}\n")
            f.write(f"TRACE_LEVEL = 0\n")
            f.write(f"TOTAL_TIME_LIMIT = {self.TOTAL_TIME_LIMIT}\n")
            f.write(f"STOP_AT_OPTIMUM = YES\n")
            f.write(f"RUNS = 10\n")
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
        return min(time, self.TOTAL_TIME_LIMIT)

    def remove_config_file(self, config_filepath: Path):
        config_filepath.unlink()
