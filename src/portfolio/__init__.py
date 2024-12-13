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
    db_instance_exists,
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
            portfolio.append(solver)
        return portfolio

    @classmethod
    def from_solver_list(cls, solvers: List[Solver]) -> "Portfolio":
        portfolio = cls()
        for solver in solvers:
            portfolio.append(solver.copy())
        return portfolio

    def append(self, solver: Solver):
        self._solvers.append(solver)

    def log(self):
        logger.debug("  Portfolio  ".center(80, "="))
        for solver in self._solvers:
            solver.log()
        logger.debug("=" * 80)

    def __len__(self):
        return len(self._solvers)

    @property
    def size(self) -> int:
        return len(self)

    def __getitem__(self, item) -> Solver:
        return self._solvers[item]

    def __setitem__(self, key, value):
        self._solvers[key] = value

    def __iter__(self):
        return iter(self._solvers)

    def copy(self) -> "Portfolio":
        return copy.deepcopy(self)

    def evaluate(
        self,
        instances: InstanceSet,
        remaining_time: np.ndarray = None,
        comment: str = "",
        calculate_instance_features: bool = False,
        cache: bool = True,
    ) -> float:
        logger.debug("executor start")

        if remaining_time is None:
            remaining_time = np.ones(shape=(self.size,)) * np.inf

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

        conn = db_connect()

        if calculate_instance_features:
            futures = []
            for instance in instances:
                if not db_instance_exists(conn, instance):
                    future = executor.submit(_calculate_instance_features, instance)
                    futures.append((instance, future))
            for instance, future in futures:
                time, features = future.result()
                remaining_time -= time
                logger.debug(
                    f"{instance.__hash__()} features calculated (time={time:.2f})"
                )
                db_insert_instance(conn, instance, features)
        else:
            for instance in instances:
                db_insert_instance(conn, instance)

        for solver in self._solvers:
            db_insert_solver(conn, solver)

        shape = (instances.size, self.size)
        max_cost = np.array([s.MAX_COST for s in self._solvers])
        costs = np.ones(shape=shape) * max_cost

        futures = np.empty(shape=shape, dtype=object)

        for i in range(instances.size):
            for j in range(self.size):
                cached_result = db_fetch_result(conn, instances[i], self._solvers[j])
                if cached_result is not None and cache:
                    logger.debug(f"({i}, {j}) cached result")
                    cost, _ = cached_result
                    costs[i, j] = cost
                elif remaining_time[j] > 0:
                    logger.debug(f"({i}, {j}) fn submitted")
                    futures[i, j] = executor.submit(
                        _solve_instance,
                        instances[i],
                        self._solvers[j],
                    )

        for i in range(instances.size):
            for j in range(self.size):
                future = futures[i, j]
                if future is None:
                    logger.debug(f"({i}, {j}) future None")
                elif remaining_time[j] <= 0:
                    logger.debug(f"({i}, {j}) no remaining time")
                    time = 0
                    future.cancel()
                else:
                    logger.debug(f"({i}, {j}) result")
                    try:
                        cost, time = future.result(timeout=13)
                    except concurrent.futures.TimeoutError:
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
