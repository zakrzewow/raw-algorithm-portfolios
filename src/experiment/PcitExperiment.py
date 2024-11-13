from typing import Type

import numpy as np

from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


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

    def construct_portfolio(self, training_instances: InstanceSet) -> Portfolio:
        portfolio = Portfolio.from_solver_class(self.solver_class, self.K)
        configuration_time = np.ones(shape=(portfolio.size,)) * self.t_c
        portfolio.evaluate(
            training_instances,
            remaining_time=configuration_time,
            calculate_instance_features=self.CALCULATE_INSTANCE_FEATURES,
        )
        return portfolio
