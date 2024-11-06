import concurrent.futures
from typing import List, Tuple, Type

import numpy as np
from ConfigSpace import ConfigurationSpace

from src.instance import Instance
from src.solver import Solver

MAX_WORKERS = 4


def _solve_instance(solver, instance):
    return solver.solve(instance)


class Portfolio:
    def __init__(
        self,
        size: int,
        solver_class: Type[Solver],
        configspace: ConfigurationSpace,
    ):
        self.size = size
        self.solvers = [
            solver_class(configspace.sample_configuration()) for _ in range(size)
        ]

    def evaluate(self, instances: List[Instance]) -> Tuple[float, float]:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
        futures = []
        for instance in instances:
            row_futures = []
            for solver in self.solvers:
                row_futures.append(executor.submit(_solve_instance, solver, instance))
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
