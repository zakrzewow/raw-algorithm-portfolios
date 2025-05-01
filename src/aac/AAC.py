import logging
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
from smac import AlgorithmConfigurationFacade, Scenario
from smac.initial_design import RandomInitialDesign
from smac.runhistory.dataclasses import TrialValue

from src.constant import SEED, TEMP_DIR
from src.instance.InstanceList import InstanceList
from src.log import logger
from src.solver.Portfolio import Portfolio
from src.surrogate.SurrogatePolicy import EmptySurrogatePolicy, SurrogatePolicy


class AAC:
    _RNG = np.random.default_rng(SEED)

    def __init__(
        self,
        portfolio: Portfolio,
        instance_list: InstanceList,
        prefix: str,
        t_c: float = np.inf,
        max_iter: int = None,
        i: int = None,
        calculate_features: bool = False,
        surrogate_policy: SurrogatePolicy = EmptySurrogatePolicy(),
    ):
        self._portfolio = portfolio
        self._instance_list = instance_list
        self._prefix = prefix
        self._t_c = t_c
        self._max_iter = max_iter
        self.iter = 1
        self._smac = self._get_smac_algorithm_configuration_facade(i)
        self._calculate_features = calculate_features
        self._surrogate_policy = surrogate_policy
        self.incumbent_cost = np.inf

    def __del__(self):
        self.__cleanup_temp_dir()

    def __repr__(self):
        str_ = f"AAC(prefix={self._prefix}, iter={self.iter}/{self._max_iter})"
        return str_

    def log(self):
        logger.debug(self.__repr__())

    def _get_smac_algorithm_configuration_facade(self, i: int):
        self.__set_temp_dir()
        scenario = Scenario(
            configspace=self._portfolio.get_configuration_space(i),
            output_directory=self.__temp_dir_path,
            deterministic=True,
            n_trials=10000,
            use_default_config=False,
            seed=self._RNG.integers(0, 2**32 - 1),
        )
        intensifier = AlgorithmConfigurationFacade.get_intensifier(
            scenario,
            max_config_calls=1,
        )
        n_configs = 0
        additional_configs = [self._portfolio.get_configuration(i)]
        initial_design = RandomInitialDesign(
            scenario,
            n_configs=n_configs,
            additional_configs=additional_configs,
        )
        smac = AlgorithmConfigurationFacade(
            scenario,
            lambda seed: None,
            initial_design=initial_design,
            intensifier=intensifier,
            logging_level=logging.CRITICAL,
            overwrite=True,
        )
        return smac

    def __set_temp_dir(self):
        self.__temp_dir = tempfile.TemporaryDirectory(dir=TEMP_DIR)
        self.__temp_dir_path = Path(self.__temp_dir.name)

    def __cleanup_temp_dir(self):
        if self.__temp_dir is not None:
            self.__temp_dir.cleanup()
            self.__temp_dir = None
            self.__temp_dir_path = None

    def configure_iter(self) -> Generator[Portfolio, None, Portfolio]:
        while self._iteration_remains():
            self.log()
            trial_info = self._smac.ask()
            self._portfolio.update_solvers(trial_info.config)
            self._surrogate_policy.notify_iter(self.iter)
            result = self._portfolio.evaluate(
                instance_list=self._instance_list,
                prefix=self._get_iteration_prefix(),
                calculate_features=self._calculate_features,
                cache=True,
                surrogate_policy=self._surrogate_policy,
            )
            if self._surrogate_policy.should_reevaluate_portfolio(
                result,
                self.incumbent_cost,
            ):
                result = self._portfolio.evaluate(
                    instance_list=self._instance_list,
                    prefix=self._get_iteration_prefix() + ";reevaluate",
                )
            trial_value = TrialValue(cost=result.cost)
            self._smac.tell(trial_info, trial_value)
            self._update_incumbent_cost(result.cost)
            yield self._portfolio
            self._next_iteration()
        self._update_portfolio_with_incumbent()
        logger.debug(f"AAC.Results(prefix={self._prefix}, portfolio={self._portfolio})")
        return self._portfolio

    def configure(self) -> Portfolio:
        for _ in self.configure_iter():
            pass
        return self._portfolio

    def update(
        self,
        instance_list: InstanceList = None,
    ):
        if instance_list is not None:
            self._instance_list = instance_list

    def _iteration_remains(self):
        return self.iter <= self._max_iter

    def _get_iteration_prefix(self):
        return f"{self._prefix};aac_iter={self.iter}"

    def _update_incumbent_cost(self, cost: float):
        if cost < self.incumbent_cost:
            self.incumbent_cost = cost

    def _next_iteration(self):
        self.iter += 1

    def _update_portfolio_with_incumbent(self):
        incumbent = self._smac.intensifier.get_incumbent()
        self._portfolio.update_solvers(incumbent)
