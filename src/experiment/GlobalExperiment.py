from typing import Type

import numpy as np

from src.experiment import Experiment
from src.instance import InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class GlobalExperiment(Experiment):
    NAME = "GLOBAL"

    def __init__(
        self,
        t_c: int,
        t_v: int,
        K: int,
        n: int,
        solver_class: Type[Solver],
    ):
        super().__init__(t_c, t_v, K, n, solver_class)

    def construct_portfolio(self, training_instances: InstanceSet) -> Portfolio:
        best_portfolio = None
        best_cost = np.inf

        for _ in range(self.n):
            logger.info(f"Attempt {_ + 1}/{self.n}")

            portfolio = Portfolio.from_solver_class(self.solver_class, self.K)
            configuration_space = portfolio.get_configuration_space()

            cost = self._configure_and_validate(
                portfolio,
                training_instances,
                configuration_space,
            )

            logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
            if cost < best_cost:
                logger.info(f"New best portfolio found!")
                best_cost = cost
                best_portfolio = portfolio
        return best_portfolio
