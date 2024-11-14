from typing import Dict, Type

import numpy as np

from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class _Clustering:
    def __init__(self, portfolio: Portfolio, instances: InstanceSet):
        self.portfolio = portfolio
        self.instances = instances
        self._cluster_assignments = self._get_random_cluster_assignments()

    def _get_random_cluster_assignments(self) -> Dict[str, int]:
        cluster_assignments = np.random.choice(self.portfolio.size, self.instances.size)
        while True:
            counts = np.bincount(cluster_assignments, minlength=self.portfolio.size)
            if max(counts) - min(counts) <= 1:
                break
            cluster_assignments = np.random.choice(
                self.portfolio.size, self.instances.size
            )

        cluster_assignments = {
            self.instances[i].__hash__(): j for i, j in enumerate(cluster_assignments)
        }
        return cluster_assignments

    def get_instances_for_solver(self, i: int):
        instance_list = [
            self.instances[j]
            for j in range(self.instances.size)
            if self._cluster_assignments[self.instances[j].__hash__()] == i
        ]
        return InstanceSet.from_instance_list(instance_list)


class PcitExperiment(Experiment):
    NAME = "PCIT"
    CALCULATE_INSTANCE_FEATURES = True

    def __init__(
        self,
        t_c: int,
        t_v: int,
        K: int,
        n: int,
        max_iter: int,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        super().__init__(t_c, t_v, K, n, solver_class, instance_class)
        self.max_iter = max_iter

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        best_portfolio = None
        best_cost = np.inf

        for _ in range(self.n):
            logger.info(f"Attempt {_ + 1}/{self.n}")
            portfolio = Portfolio.from_solver_class(self.solver_class, self.K)
            clustering = _Clustering(portfolio, train_instances)
            logger.info(f"Clustering: {clustering._cluster_assignments}")

            for i in range(self.K):
                logger.info(f"Solver {i + 1}/{self.K}")
                solver = portfolio[i]
                temp_portfolio = Portfolio.from_solver_list([solver])
                temp_instances = clustering.get_instances_for_solver(i)
                configuration_space = temp_portfolio.get_configuration_space()
                self._configure_and_validate(
                    temp_portfolio,
                    temp_instances,
                    configuration_space,
                )
                portfolio[i] = temp_portfolio[0]

            cost = self._validate(portfolio, train_instances)
            logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
            if cost < best_cost:
                logger.info(f"New best portfolio found!")
                best_cost = cost
                best_portfolio = portfolio
        return best_portfolio
