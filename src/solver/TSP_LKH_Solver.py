import os
import subprocess
from pathlib import Path

from ConfigSpace import Configuration

from src.configuration_space.LKH import CONFIGURATION_SPACE
from src.constant import LKH_PATH, SEED, TEMP_DIR
from src.instance import TSP_Instance
from src.solver.Solver import Solver


class TSP_LKH_Solver(Solver):
    CONFIGURATION_SPACE = CONFIGURATION_SPACE
    MAX_COST = 100.0
    MAX_TIME = 10.0

    def __init__(self, config: Configuration = None):
        super().__init__(config)

    @classmethod
    def _solve(
        cls,
        prefix: str,
        solver: "TSP_LKH_Solver",
        instance: TSP_Instance,
        features_time: float = 0.0,
    ) -> Solver.Result:
        config_filepath = solver._to_config_file(instance)
        try:
            result = subprocess.run(
                [LKH_PATH, config_filepath],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=solver.MAX_TIME + 5,
            )
            with open("tmp.log", "w") as log_file:
                log_file.write(result.stdout)
            time = solver._parse_result(result)
            cost = time if time < solver.MAX_TIME else solver.MAX_COST
            error = False
        except subprocess.TimeoutExpired:
            time = solver.MAX_TIME
            cost = solver.MAX_COST
            error = True
        time += features_time
        solver._remove_config_file(config_filepath)
        return Solver.Result(prefix, solver, instance, cost, time, error=error)

    def _to_config_file(self, instance: TSP_Instance) -> Path:
        config_filepath = TEMP_DIR / f"config_{os.getpid()}.par"
        with open(config_filepath, "w") as f:
            f.write(f"PROBLEM_FILE = {instance.filepath}\n")
            f.write(f"OPTIMUM = {instance.optimum}\n")
            f.write(f"TRACE_LEVEL = 1000\n")
            f.write(f"TOTAL_TIME_LIMIT = {self.MAX_TIME}\n")
            f.write(f"TIME_LIMIT = {self.MAX_TIME}\n")
            f.write(f"STOP_AT_OPTIMUM = YES\n")
            f.write(f"RUNS = 10000\n")
            f.write(f"SEED = {SEED}\n")
            for k, v in self.config.items():
                f.write(f"{k} = {v}\n")
        return config_filepath

    def _parse_result(self, result: subprocess.CompletedProcess) -> float:
        time = None
        for line in result.stdout.splitlines():
            if "Time.total" in line:
                time = float(line.split()[-2])
                break
        if time is None:
            raise Exception("Time.total not found")
        return min(time, self.MAX_TIME)

    def _remove_config_file(self, config_filepath: Path):
        config_filepath.unlink(missing_ok=True)
