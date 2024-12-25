import logging
import tempfile
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario
from smac.initial_design import RandomInitialDesign
from smac.runhistory.dataclasses import TrialValue

from src.constant import TEMP_DIR
from src.instance.InstanceList import InstanceList
from src.log import logger
from src.solver.Portfolio import Portfolio


class AAC:
    def __init__(
        self,
        prefix: str,
        portfolio: Portfolio,
        i: int = None,
        is_random_initial: bool = False,
    ):
        self.prefix = prefix
        self.portfolio = portfolio
        self._configuration_space = self.portfolio.get_configuration_space(i)
        self.is_random_initial = is_random_initial
        self._iter = 0

    def __del__(self):
        self.__cleanup_temp_dir()

    def _get_smac_algorithm_configuration_facade(self):
        self.__set_temp_dir()
        scenario = Scenario(
            configspace=self._configuration_space,
            output_directory=self.__temp_dir_path,
            use_default_config=False,
            deterministic=True,
            seed=-1,
        )
        intensifier = AlgorithmConfigurationFacade.get_intensifier(
            scenario,
            max_config_calls=1,
        )
        if initial_configuration is not None:
            n_configs = 0
            additional_configs = [initial_configuration]
        else:
            n_configs = 1
            additional_configs = []
        initial_design = RandomInitialDesign(
            scenario,
            n_configs=n_configs,
            additional_configs=additional_configs,
        )
        smac = AlgorithmConfigurationFacade(
            scenario,
            lambda seed: None,
            overwrite=True,
            logging_level=logging.CRITICAL,
            intensifier=intensifier,
            initial_design=initial_design,
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

    # def _configure_and_validate(
    #     self,
    #     instance_list: InstanceList,
    # ) -> float:
    #     incumbent = self._configure_wtih_smac(
    #         portfolio,
    #         train_instances,
    #         configuration_space,
    #         initial_configuration,
    #     )
    #     portfolio.update_config(incumbent)
    #     cost = self._validate(portfolio, train_instances)
    #     return cost

    # def _configure_wtih_smac(
    #     self,
    #     portfolio: Portfolio,
    #     train_instances: InstanceSet,
    #     configuration_space: ConfigurationSpace,
    #     initial_configuration: Configuration = None,
    # ) -> Configuration:
    #     smac = self._get_smac_algorithm_configuration_facade(
    #         configuration_space,
    #         initial_configuration,
    #     )
    #     configuration_time = np.ones(shape=(portfolio.size,)) * self.t_c
    #     logger.debug(f"SMAC configuration, time: {configuration_time}")
    #     iteration = 1
    #     while (configuration_time > 0).all():
    #         trial_info = smac.ask()
    #         portfolio.update_config(trial_info.config)
    #         logger.debug(
    #             f"SMAC iteration {iteration}, configuration: {dict(trial_info.config)}"
    #         )
    #         cost = portfolio.evaluate(
    #             train_instances,
    #             configuration_time,
    #             "configuration",
    #             calculate_instance_features=self.CALCULATE_INSTANCE_FEATURES,
    #         )
    #         logger.debug(
    #             f"SMAC iteration {iteration}, cost: {cost:.2f}, configuration time: {configuration_time}"
    #         )
    #         trial_value = TrialValue(cost=cost)
    #         smac.tell(trial_info, trial_value)
    #         iteration += 1
    #     incumbent = smac.intensifier.get_incumbent()
    #     return incumbent
