import concurrent.futures
from typing import List, Type

import numpy as np
from ConfigSpace import Configuration

from src.constant import MAX_WORKERS
from src.instance import Instance
from src.solver import Solver


def _solve_instance(solver: Solver, instance: Instance) -> float:
    return solver.solve(instance)


class Portfolio:
    def __init__(
        self,
        size: int,
        solver_class: Type[Solver],
    ):
        self.size = size
        self.solvers = [solver_class() for _ in range(size)]

    def evaluate(
        self,
        instances: List[Instance],
        remaining_time: np.ndarray,
        max_cost: float,
    ) -> float:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
        n_instances = len(instances)
        futures = np.empty(shape=(n_instances, self.size), dtype=object)
        for i in range(n_instances):
            for j in range(self.size):
                if remaining_time[j] > 0:
                    futures[i, j] = executor.submit(
                        _solve_instance, self.solvers[j], instances[i]
                    )

        costs = np.ones(shape=(n_instances, self.size)) * max_cost
        for i in range(n_instances):
            for j in range(self.size):
                future = futures[i, j]
                if future is None:
                    continue
                elif remaining_time[j] <= 0:
                    future.cancel()
                else:
                    cost, time = future.result()
                    remaining_time[j] -= time
                    costs[i, j] = cost

        executor.shutdown(cancel_futures=True)
        cost = costs.min(axis=1).mean()
        return cost

    def update_config(self, config: Configuration):
        for k, v in config.items():
            idx, key = k.split("__")
            idx = int(idx)
            self.solvers[idx].config[key] = v
