from typing import Type

import numpy as np
from ConfigSpace import Configuration

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
        K: int,
        n: int,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        super().__init__(t_c, K, n, solver_class, instance_class)

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        solvers = []
        largest_marginal_contribution_solver = None
        for i in range(self.K):
            logger.info(f"Solver {i + 1}/{self.K}")

            best_cost = np.inf
            best_solver = None
            attempt_solvers = []
            for _ in range(self.n):
                logger.info(f"Attempt {_ + 1}/{self.n}")

                if largest_marginal_contribution_solver is not None:
                    new_solver = largest_marginal_contribution_solver.copy()
                else:
                    new_solver = self.solver_class()
                iteration_solvers = solvers + [new_solver]
                portfolio = Portfolio.from_solver_list(iteration_solvers)
                configuration_space = portfolio.get_configuration_space(i=i)
                config = {f"{i}__{k}": v for k, v in new_solver.config.items()}
                config = Configuration(configuration_space, values=config)
                cost = self._configure_and_validate(
                    portfolio,
                    train_instances,
                    configuration_space,
                    config,
                )
                attempt_solvers.append(portfolio[i])
                logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
                if cost < best_cost:
                    best_cost = cost
                    best_solver = portfolio[i]
            solvers.append(best_solver)
            logger.info(f"Solver {i + 1}/{self.K} - best cost = {best_cost:.2f}")

            if i < self.K - 1:
                largest_marginal_contribution_solver = None
                best_cost = np.inf
                for solver in attempt_solvers:
                    if solver != best_solver:
                        portfolio = Portfolio.from_solver_list(solvers + [solver])
                        cost = portfolio.evaluate(
                            train_instances,
                            comment="largest_marginal_contribution",
                        )
                        if cost < best_cost:
                            best_cost = cost
                            largest_marginal_contribution_solver = solver

        portfolio = Portfolio.from_solver_list(solvers)
        return portfolio
