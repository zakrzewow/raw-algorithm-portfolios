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
                timeout=instance.cut_off_time + 5,
            )
            time = solver._parse_result(result, instance)
            cost = time if time < instance.cut_off_time else instance.cut_off_cost
            error = False
        except subprocess.TimeoutExpired:
            time = instance.cut_off_time
            cost = instance.cut_off_cost
            error = True
        time += features_time
        solver._remove_config_file(config_filepath)
        return Solver.Result(prefix, solver, instance, cost, time, error=error)

    def _to_config_file(self, instance: TSP_Instance) -> Path:
        config_filepath = TEMP_DIR / f"config_{os.getpid()}.par"
        with open(config_filepath, "w") as f:
            f.write(f"PROBLEM_FILE = {instance.filepath}\n")
            f.write(f"OPTIMUM = {instance.optimum}\n")
            f.write(f"TRACE_LEVEL = 0\n")
            f.write(f"TOTAL_TIME_LIMIT = {instance.cut_off_time}\n")
            f.write(f"TIME_LIMIT = {instance.cut_off_time}\n")
            f.write(f"STOP_AT_OPTIMUM = YES\n")
            f.write(f"RUNS = 10000\n")
            f.write(f"SEED = {SEED}\n")
            for k, v in self.config.items():
                f.write(f"{k} = {v}\n")
        return config_filepath

    def _parse_result(
        self,
        result: subprocess.CompletedProcess,
        instance: TSP_Instance,
    ) -> float:
        time = None
        for line in result.stdout.splitlines():
            if "Time.total" in line:
                time = float(line.split()[-2])
                break
        if time is None:
            raise Exception("Time.total not found")
        return min(time, instance.cut_off_time)

    def _remove_config_file(self, config_filepath: Path):
        config_filepath.unlink(missing_ok=True)
