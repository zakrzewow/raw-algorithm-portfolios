import logging
import tempfile
from pathlib import Path

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
        prefix: str,
        portfolio: Portfolio,
        t_c: float = None,
        max_iter: int = None,
        i: int = None,
        calculate_features: bool = False,
    ):
        self.prefix = prefix
        self.portfolio = portfolio
        self._i = i
        t_c = np.inf if t_c is None else t_c
        self._configuration_time = self._get_configuration_time(t_c)
        self._iter = 1
        self._max_iter = np.inf if max_iter is None else max_iter
        self._configuration_space = self.portfolio.get_configuration_space(i)
        self._calculate_features = calculate_features
        self._smac = self._get_smac_algorithm_configuration_facade()

    def __del__(self):
        self.__cleanup_temp_dir()

    def __repr__(self):
        time_formatted = np.array2string(
            self._configuration_time,
            precision=2,
            floatmode="fixed",
        )
        str_ = f"AAC(prefix={self.prefix}, iter={self._iter}, configuration_time={time_formatted})"
        return str_

    def log(self):
        logger.debug(self.__repr__())

    def _get_configuration_time(self, t_c: float):
        configuration_time = np.ones(shape=(self.portfolio.size,)) * t_c
        return configuration_time

    def _get_smac_algorithm_configuration_facade(self):
        self.__set_temp_dir()
        scenario = Scenario(
            configspace=self._configuration_space,
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
        additional_configs = [self.portfolio.get_configuration(self._i)]
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

    def configure(
        self,
        instance_list: InstanceList,
        estimator: BaseEstimator = None,
    ) -> Portfolio:
        while self._configuration_time_remains():
            self.log()
            trial_info = self._smac.ask()
            self.portfolio.update_solvers(trial_info.config)
            result = self.portfolio.evaluate(
                instance_list=instance_list,
                prefix=self._get_iteration_prefix(),
                calculate_features=self._calculate_features,
                cache=True,
                estimator=estimator,
            )
            trial_value = TrialValue(cost=result.cost)
            self._smac.tell(trial_info, trial_value)
            self._update_configuration_time(result.time)
            self._next_iteration()
        self._update_portfolio_with_incumbent()
        logger.debug(f"AAC.Results(prefix={self.prefix}, portfolio={self.portfolio})")
        return self.portfolio

    def _configuration_time_remains(self):
        return (self._configuration_time > 0).all() and self._iter <= self._max_iter

    def _get_iteration_prefix(self):
        return f"{self.prefix};aac_iter={self._iter}"

    def _update_configuration_time(self, time: np.array):
        self._configuration_time -= time

    def _next_iteration(self):
        self._iter += 1

    def _update_portfolio_with_incumbent(self):
        incumbent = self._smac.intensifier.get_incumbent()
        self.portfolio.update_solvers(incumbent)
