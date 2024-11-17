import random
from typing import Type

import numpy as np
from ConfigSpace import Configuration

from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class CepsExperiment(Experiment):
    NAME = "CEPS"

    def __init__(
        self,
        t_c: int,
        t_v: int,
        t_ini: int,
        t_i: int,
        K: int,
        n: int,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        super().__init__(t_c, t_v, K, n, solver_class, instance_class)
        self.t_ini = t_ini
        self.t_i = t_i

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        portfolio = self._initialization(train_instances)
        best_cost = portfolio.evaluate(
            train_instances,
            np.ones(shape=(self.K,)) * np.inf,
            comment="initialization_validation",
        )
        for phase in range(self.max_iter):
            logger.info(f"Phase {phase + 1}/{self.max_iter}")
            for _ in range(self.n):
                logger.info(f"Attempt {_ + 1}/{self.n}")
                i = random.choice(range(portfolio.size))
                temp_portfolio = Portfolio.copy()
                temp_portfolio[i] = self.solver_class()
                configuration_space = portfolio.get_configuration_space(i=i)
                config = {f"{i}__{k}": v for k, v in temp_portfolio[i].config.items()}
                config = Configuration(configuration_space, values=config)
                cost = self._configure_and_validate(
                    temp_portfolio,
                    train_instances,
                    configuration_space,
                    config,
                )
                logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
                if cost < best_cost:
                    best_cost = cost
                    portfolio = temp_portfolio

        # for
        return portfolio

    def _initialization(self, train_instances: InstanceSet) -> Portfolio:
        solvers = []
        logger.info("Initialization")

        num_of_configs = self.t_ini // sum(
            [i * train_instances.size * 10 for i in range(1, self.K + 1)]
        )
        configuration_list = [
            self.solver_class.CONFIGURATION_SPACE.sample_configuration()
            for _ in range(num_of_configs)
        ]
        for config in configuration_list:
            logger.debug(f"{dict(config)}")
        for i in range(self.K):
            logger.info(f"Solver {i + 1}/{self.K}")

            best_cost = np.inf
            best_solver = None
            for ic, config in enumerate(configuration_list):
                logger.info(f"Random config {ic + 1}/{num_of_configs}")
                new_solver = self.solver_class(config)
                iteration_solvers = solvers + [new_solver]
                portfolio = Portfolio.from_solver_list(iteration_solvers)
                portfolio.log()
                time = np.ones(len(iteration_solvers)) * train_instances.size * 10
                cost = portfolio.evaluate(
                    train_instances,
                    time,
                    comment="initialization",
                )
                logger.info(
                    f"Random config {ic + 1}/{num_of_configs}: cost = {cost:.2f}"
                )
                if cost < best_cost:
                    best_cost = cost
                    best_solver = portfolio[i]
            solvers.append(best_solver)
            logger.info(f"Solver {i + 1}/{self.K} - best cost = {best_cost:.2f}")
        portfolio = Portfolio.from_solver_list(solvers)
        logger.info("Initialization done")
        portfolio.log()
        return portfolio
