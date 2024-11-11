import concurrent.futures
import copy
from typing import List, Type

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from src.constant import MAX_WORKERS
from src.instance import Instance, InstanceSet
from src.solver import Solver


def _solve_instance(solver: Solver, instance: Instance) -> float:
    return solver.solve(instance)


class Portfolio:
    def __init__(
        self,
    ):
        self._solvers = []

    @classmethod
    def from_solver_class(cls, solver_class: Type[Solver], size: int) -> "Portfolio":
        portfolio = cls()
        for _ in range(size):
            solver = solver_class()
            portfolio.add_solver(solver)
        return portfolio

    @classmethod
    def from_solver_list(cls, solvers: List[Solver]) -> "Portfolio":
        portfolio = cls()
        for solver in solvers:
            portfolio.add_solver(solver.copy())
        return portfolio

    def add_solver(self, solver: Solver):
        self._solvers.append(solver)

    @property
    def size(self) -> int:
        return len(self._solvers)

    def evaluate(
        self,
        instances: InstanceSet,
        remaining_time: np.ndarray,
    ) -> float:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
        n_instances = len(instances)
        shape = (n_instances, self.size)
        futures = np.empty(shape=shape, dtype=object)
        for i in range(n_instances):
            for j in range(self.size):
                if remaining_time[j] > 0:
                    futures[i, j] = executor.submit(
                        _solve_instance,
                        self._solvers[j],
                        instances[i],
                    )

        max_cost = np.array([s.MAX_COST for s in self._solvers])
        costs = np.ones(shape=shape) * max_cost
        for i in range(n_instances):
            for j in range(self.size):
                future = futures[i, j]
                if future is None:
                    pass
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
            self._solvers[idx].config[key] = v

    def get_configuration_space(self, i: int = None) -> ConfigurationSpace:
        configuration_space = ConfigurationSpace()
        for idx, solver in enumerate(self._solvers):
            if i is not None and idx != i:
                continue
            for _, v in solver.config.config_space.items():
                v = copy.deepcopy(v)
                v.name = f"{idx}__{v.name}"
                configuration_space.add(v)
        return configuration_space
