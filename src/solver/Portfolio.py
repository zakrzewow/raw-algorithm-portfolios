import concurrent.futures
import copy
from typing import Iterable, Type

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.base import BaseEstimator

from src.constant import MAX_WORKERS, SEED
from src.instance.InstanceList import InstanceList
from src.log import logger
from src.solver.Solver import Solver


class Portfolio(list):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_iterable(cls, solvers: Iterable[Solver]) -> "Portfolio":
        portfolio = cls()
        portfolio.extend(solvers)
        return portfolio

    @classmethod
    def from_solver_class(cls, solver_class: Type[Solver], size: int) -> "Portfolio":
        portfolio = cls()
        for _ in range(size):
            solver = solver_class()
            portfolio.append(solver)
        return portfolio

    def __repr__(self):
        str_ = super().__repr__()
        str_ = f"Portfolio(size={self.size}){str_}"
        return str_

    def copy(self) -> "Portfolio":
        return copy.deepcopy(self)

    def log(self):
        logger.debug(self.__repr__())

    @property
    def size(self):
        return len(self)

    def update_solvers(self, config: Configuration):
        for k, v in config.items():
            idx, key = k.split("__")
            idx = int(idx)
            self[idx].config[key] = v

    def get_configuration_space(self, i: int = None) -> ConfigurationSpace:
        configuration_space = ConfigurationSpace()
        for idx, solver in enumerate(self):
            if i is not None and idx != i:
                continue
            for _, v in solver.config.config_space.items():
                v = copy.deepcopy(v)
                v.name = f"{idx}__{v.name}"
                configuration_space.add(v)
        return configuration_space

    def get_configuration(self, i: int = None) -> Configuration:
        configuration_space = self.get_configuration_space(i=i)
        config = {}
        for idx, solver in enumerate(self):
            if i is not None and idx != i:
                continue
            for k, v in solver.config.items():
                config[f"{idx}__{k}"] = v
        config = Configuration(configuration_space, values=config)
        return config

    class Result:
        def __init__(
            self,
            portfolio: "Portfolio",
            instance_list: InstanceList,
            prefix: str = "",
        ):
            self.prefix = prefix
            shape = (instance_list.size, portfolio.size)
            self._costs = np.zeros(shape)
            self.time = np.zeros(portfolio.size)
            self.instance_sover_to_idx = {}
            for i, instance in enumerate(instance_list):
                for j, solver in enumerate(portfolio):
                    key = (instance.id(), solver.id())
                    self.instance_sover_to_idx[key] = (i, j)

        def __repr__(self):
            time_formatted = np.array2string(self.time, precision=2, floatmode="fixed")
            str_ = f"Portfolio.Result(prefix={self.prefix}, cost={self.cost:.2f}, time={time_formatted})"
            return str_

        def log(self):
            logger.debug(self.__repr__())

        @property
        def cost(self):
            return self._costs.min(axis=1).mean()

        def update(self, result: Solver.Result):
            key = (result.instance.id(), result.solver.id())
            i, j = self.instance_sover_to_idx[key]
            self._costs[i, j] = result.cost
            self.time[j] += result.time

    def evaluate(
        self,
        instance_list: InstanceList,
        prefix: str,
        calculate_features: bool = False,
        cache: bool = True,
        estimator: BaseEstimator = None,
        estimator_pct: float = 0.9,
    ) -> Result:
        logger.debug(f"Portfolio.evaluate({prefix})")
        self.log()

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
        rng = np.random.default_rng(SEED)
        result = self.Result(self, instance_list, prefix=prefix)
        futures = []
        for instance in instance_list:
            for solver in self:
                do_use_estimator = rng.random() < estimator_pct
                future = solver.solve(
                    instance,
                    prefix,
                    calculate_features=calculate_features,
                    cache=cache,
                    estimator=(estimator if do_use_estimator else None),
                    executor=executor,
                )
                futures.append((instance, solver, future))

        for instance, solver, future in futures:
            try:
                solver_result = future.result(timeout=solver.MAX_TIME + 10)
            except concurrent.futures.TimeoutError:
                future.cancel()
                solver_result = Solver.Result.error_instance(prefix, solver, instance)
                solver_result.log()
                solver_result.to_db()
            result.update(solver_result)

        executor.shutdown(wait=False, cancel_futures=True)
        result.log()
        return result
