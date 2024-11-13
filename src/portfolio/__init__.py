import concurrent.futures
import copy
from typing import Dict, List, Type

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from src.constant import MAX_WORKERS
from src.database import (
    db_connect,
    db_fetch_result,
    db_insert_instance,
    db_insert_result,
    db_insert_solver,
)
from src.instance import Instance, InstanceSet
from src.log import logger
from src.solver import Solver


def _solve_instance(instance: Instance, solver: Solver) -> float:
    return solver.solve(instance)


def _calculate_instance_features(instance: Instance) -> Dict:
    return instance.calculate_features()


class Portfolio:
    def __init__(
        self,
    ):
        self._solvers: List[Solver] = []

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
        comment: str = "",
        calculate_instance_features: bool = False,
    ) -> float:
        logger.debug("executor start")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

        conn = db_connect()

        if calculate_instance_features:
            futures = []
            for i, features in enumerate(
                executor.map(_calculate_instance_features, instances)
            ):
                logger.debug(f"{instances[i].__hash__()} features calculated")
                db_insert_instance(conn, instances[i], features)
        else:
            for instance in instances:
                db_insert_instance(conn, instance)

        for solver in self._solvers:
            db_insert_solver(conn, solver)

        n_instances = len(instances)
        shape = (n_instances, self.size)
        max_cost = np.array([s.MAX_COST for s in self._solvers])
        costs = np.ones(shape=shape) * max_cost

        futures = np.empty(shape=shape, dtype=object)

        for i in range(n_instances):
            for j in range(self.size):
                cached_result = db_fetch_result(conn, instances[i], self._solvers[j])
                if cached_result is not None:
                    logger.debug(f"({i}, {j}) cached result")
                    cost, time = cached_result
                    costs[i, j] = cost
                    remaining_time[j] = max(0, remaining_time[j] - time)
                elif remaining_time[j] > 0:
                    logger.debug(f"({i}, {j}) fn submitted")
                    futures[i, j] = executor.submit(
                        _solve_instance,
                        instances[i],
                        self._solvers[j],
                    )

        for i in range(n_instances):
            for j in range(self.size):
                future = futures[i, j]
                if future is None:
                    logger.debug(f"({i}, {j}) future None")
                    continue
                elif remaining_time[j] <= 0:
                    logger.debug(f"({i}, {j}) no remaining time")
                    time = 0
                    future.cancel()
                else:
                    logger.debug(f"({i}, {j}) result")
                    try:
                        cost, time = future.result(timeout=13)
                    except concurrent.futures.TimeoutError:
                        logger.error(
                            f"timeout: instance {instances[i].__hash__()}, solver {self._solvers[j].__hash__()}"
                        )
                        future.cancel()
                        cost, time = self._solvers[j].max_cost_time
                    remaining_time[j] = max(0, remaining_time[j] - time)
                    costs[i, j] = cost
                    db_insert_result(
                        conn,
                        instances[i],
                        self._solvers[j],
                        costs[i, j],
                        time,
                        comment,
                    )
                    logger.debug(f"({i}, {j}) result inserted")

        executor.shutdown(wait=False, cancel_futures=True)
        logger.debug("executor shutdown")
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
