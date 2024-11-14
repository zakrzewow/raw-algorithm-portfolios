from typing import Type

import numpy as np

from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class ParhydraExperiment(Experiment):
    NAME = "PARHYDRA"

    def __init__(
        self,
        t_c: int,
        t_v: int,
        K: int,
        n: int,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        super().__init__(t_c, t_v, K, n, solver_class, instance_class)

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        solvers = []
        for i in range(self.K):
            logger.info(f"Solver {i + 1}/{self.K}")
            best_solver = None
            best_cost = np.inf
            for _ in range(self.n):
                logger.info(f"Attempt {_ + 1}/{self.n}")

                new_solver = self.solver_class()
                iteration_solvers = solvers + [new_solver]
                portfolio = Portfolio.from_solver_list(iteration_solvers)
                configuration_space = portfolio.get_configuration_space(i=i)

                cost = self._configure_and_validate(
                    portfolio,
                    train_instances,
                    configuration_space,
                )

                logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
                if cost < best_cost:
                    logger.info(f"New best solver found!")
                    best_cost = cost
                    best_solver = new_solver
            solvers.append(best_solver)
        portfolio = Portfolio.from_solver_list(solvers)
        return portfolio
