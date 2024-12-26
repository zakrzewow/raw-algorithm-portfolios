import logging
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
from sklearn.base import BaseEstimator
from smac import AlgorithmConfigurationFacade, Scenario
from smac.initial_design import RandomInitialDesign
from smac.runhistory.dataclasses import TrialValue

from src.constant import SEED, TEMP_DIR
from src.instance.InstanceList import InstanceList
from src.log import logger
from src.solver.Portfolio import Portfolio


class AAC:

    def __init__(
        self,
        portfolio: Portfolio,
        instance_list: InstanceList,
        prefix: str,
        t_c: float = None,
        max_iter: int = None,
        i: int = None,
        calculate_features: bool = False,
        estimator: BaseEstimator = None,
        estimator_pct: float = 0.9,
    ):
        self.portfolio = portfolio
        self.instance_list = instance_list
        self.prefix = prefix
        t_c = np.inf if t_c is None else t_c
        self.configuration_time = self._get_configuration_time(t_c)
        self._max_iter = np.inf if max_iter is None else max_iter
        self.iter = 1
        self._smac = self._get_smac_algorithm_configuration_facade(i)
        self._calculate_features = calculate_features
        self.estimator = estimator
        self.estimator_pct = estimator_pct

    def __del__(self):
        self.__cleanup_temp_dir()

    def __repr__(self):
        time_formatted = np.array2string(
            self.configuration_time,
            precision=2,
            floatmode="fixed",
        )
        str_ = f"AAC(prefix={self.prefix}, iter={self.iter}, configuration_time={time_formatted})"
        return str_

    def log(self):
        logger.debug(self.__repr__())

    def _get_configuration_time(self, t_c: float):
        configuration_time = np.ones(shape=(self.portfolio.size,)) * t_c
        return configuration_time

    def _get_smac_algorithm_configuration_facade(self, i: int):
        self.__set_temp_dir()
        scenario = Scenario(
            configspace=self.portfolio.get_configuration_space(i),
            output_directory=self.__temp_dir_path,
            deterministic=True,
            n_trials=10000,
            use_default_config=False,
            seed=SEED,
        )
        intensifier = AlgorithmConfigurationFacade.get_intensifier(
            scenario,
            max_config_calls=1,
        )
        n_configs = 0
        additional_configs = [self.portfolio.get_configuration(i)]
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

    def configure(self) -> Generator["AAC", None, "AAC"]:
        while self._configuration_time_remains():
            self.log()
            trial_info = self._smac.ask()
            self.portfolio.update_solvers(trial_info.config)
            result = self.portfolio.evaluate(
                instance_list=self.instance_list,
                prefix=self._get_iteration_prefix(),
                calculate_features=self._calculate_features,
                cache=True,
                estimator=self.estimator,
                estimator_pct=self.estimator_pct,
            )
            trial_value = TrialValue(cost=result.cost)
            self._smac.tell(trial_info, trial_value)
            self._update_configuration_time(result.time)
            yield self
            self._next_iteration()
        self._update_portfolio_with_incumbent()
        logger.debug(f"AAC.Results(prefix={self.prefix}, portfolio={self.portfolio})")
        return self

    def _configuration_time_remains(self):
        return (self.configuration_time > 0).all() and self.iter <= self._max_iter

    def _get_iteration_prefix(self):
        return f"{self.prefix};aac_iter={self.iter}"

    def _update_configuration_time(self, time: np.array):
        self.configuration_time -= time

    def _next_iteration(self):
        self.iter += 1

    def _update_portfolio_with_incumbent(self):
        incumbent = self._smac.intensifier.get_incumbent()
        self.portfolio.update_solvers(incumbent)
