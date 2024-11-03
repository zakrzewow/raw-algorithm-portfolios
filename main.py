import concurrent.futures
import datetime as dt
import multiprocessing
import os
import subprocess
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    GreaterThanCondition,
    Integer,
)
from smac import HyperparameterOptimizationFacade, Scenario

MAX_WORKERS = 4
LKH_CONFIGURATION_SPACE = ConfigurationSpace(
    seed=0,
    space=[
        Integer("ASCENT_CANDIDATES", (40, 60), default=50),
        Integer("BACKBONE_TRIALS", (0, 1), default=0),
        Categorical("BACKTRACKING", ["YES", "NO"], default="NO"),
        Categorical("CANDIDATE_SET_TYPE", ["ALPHA", "DELAUNAY", "NEAREST-NEIGHBOR", "QUADRANT"], default="ALPHA"),
        Integer("EXTRA_CANDIDATES", (0, 10), default=0),
        # Categorical("EXTRA_CANDIDATE_SET_TYPE", ["NEAREST-NEIGHBOR", "QUADRANT"], default="QUADRANT"),
        Categorical("EXTRA_CANDIDATE_SET_TYPE", ["QUADRANT"], default="QUADRANT"),
        Categorical("GAIN23", ["YES", "NO"], default="YES"),
        Categorical("GAIN_CRITERION", ["YES", "NO"], default="YES"),
        Integer("INITIAL_STEP_SIZE", (1, 5), default=1),
        Categorical("INITIAL_TOUR_ALGORITHM", ["BORUVKA", "GREEDY", "NEAREST-NEIGHBOR", "QUICK-BORUVKA", "SIERPINSKI", "WALK"], default="WALK"),
        Float("INITIAL_TOUR_FRACTION", (0, 1), default=1),
        Integer("KICKS", (0, 5), default=1),
        Categorical("KICK_TYPE", [0, 4, 5], default=0),
        Integer("MAX_BREADTH", (1, 2147483647), default=2147483647),
        Integer("MAX_CANDIDATES", (1, 10), default=5),
        Integer("MOVE_TYPE", (2, 6), default=5),
        Integer("PATCHING_A", (0, 5), default=1),
        Integer("PATCHING_C", (0, 5), default=0),
        Integer("POPULATION_SIZE", (2, 100), default=50),
        Categorical("RESTRICTED_SEARCH", ["YES", "NO"], default="YES"),
        Categorical("SUBGRADIENT", ["YES", "NO"], default="YES"),
        Categorical("SUBSEQUENT_MOVE_TYPE", [0, 2, 3, 4, 5, 6], default=0),
        Categorical("SUBSEQUENT_PATCHING", ["YES", "NO"], default="YES"),
    ],
)
# LKH_CONFIGURATION_SPACE.add(GreaterThanCondition(LKH_CONFIGURATION_SPACE["EXTRA_CANDIDATE_SET_TYPE"], LKH_CONFIGURATION_SPACE["EXTRA_CANDIDATES"], 0))


class Instance(ABC):
    pass


class TSP_Instance(Instance):
    def __init__(self, filepath: str, optimum: float):
        self.filepath = filepath
        self.optimum = optimum


class Solver(ABC):
    def __init__(self, config: Configuration):
        self.config = config

    @abstractmethod
    def solve(self, instance: Instance) -> Tuple[float, float]:
        pass

    @property
    def pid(self) -> int:
        return os.getpid()


class TSP_Solver(Solver):
    TOTAL_TIME_LIMIT = 10.0

    def __init__(self, config: Configuration):
        super().__init__(config)

    def solve(self, instance: TSP_Instance) -> Tuple[float, float]:
        print(f"[{self}][{self.pid}][{dt.datetime.now()}] Solving instance {instance.filepath}")
        config_filepath = self._to_config_file(instance.filepath, instance.optimum)
        result = subprocess.run(["LKH-2.0.10\LKH.exe", config_filepath], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        time = self._parse_result(result)
        for line in result.stdout.splitlines():
            print(line)
        cost = time if time < self.TOTAL_TIME_LIMIT else time * 10
        self.remove_config_file(config_filepath)
        return cost, time

    def _to_config_file(self, problem_filepath: str, optimum: float) -> str:
        config_filepath = f"config_{os.getpid()}.par"
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
            raise Exception("Time not found")
        return min(time, self.TOTAL_TIME_LIMIT)

    def remove_config_file(self, config_filepath: str):
        os.remove(config_filepath)


def solve_instance(solver, instance):
    return solver.solve(instance)


class Portfolio(ABC):
    _SOLVER_CLASS = Solver

    def __init__(self, size: int, configspace: ConfigurationSpace):
        self.size = size
        self.solvers = [self._SOLVER_CLASS(configspace.sample_configuration()) for _ in range(size)]

    def evaluate(self, instances: List[Instance]) -> Tuple[float, float]:
        print(f"[{dt.datetime.now()}] Evaluating {len(instances)} instances")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
        futures = []
        for instance in instances:
            row_futures = []
            for solver in self.solvers:
                row_futures.append(executor.submit(solve_instance, solver, instance))
            futures.append(row_futures)

        total_costs = []
        total_times = []
        for row_futures in futures:
            results = [future.result() for future in row_futures]
            costs, times = zip(*results)
            cost, time = min(costs), sum(times)
            total_costs.append(cost)
            total_times.append(time)
        executor.shutdown()
        return np.mean(total_costs), sum(total_times)


class TSP_Portfolio(Portfolio, ABC):
    _SOLVER_CLASS = TSP_Solver


class TSP_GlobalPortfolio(TSP_Portfolio):
    pass


if __name__ == "__main__":
    training_instances = [
        TSP_Instance("1.tsp", 20887545.00),
        TSP_Instance("2.tsp", 21134211.00),
        TSP_Instance("3.tsp", 21196547.00),
        TSP_Instance("4.tsp", 21428037.00),
        TSP_Instance("5.tsp", 11020488.00),
    ]

    portfolio = TSP_GlobalPortfolio(size=2, configspace=LKH_CONFIGURATION_SPACE)
    cost, time = portfolio.evaluate(training_instances)
    print(cost, time)
