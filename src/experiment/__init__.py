import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue

from src.constant import TEMP_DIR
from src.database import db_init
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class Experiment(ABC):
    NAME = "EXPERIMENT"
    CALCULATE_INSTANCE_FEATURES = False

    def __init__(
        self,
        t_c: int,
        t_v: int,
        K: int,
        n: int,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        self.t_c = t_c
        self.t_v = t_v
        self.K = K
        self.n = n
        self.solver_class = solver_class
        logger.info(f"[{self.NAME}] Start!")
        db_init(solver_class, instance_class, self.CALCULATE_INSTANCE_FEATURES)

    @abstractmethod
    def construct_portfolio(self, training_instances: InstanceSet) -> Portfolio:
        pass

    def _configure_and_validate(
        self,
        portfolio: Portfolio,
        training_instances: InstanceSet,
        configuration_space: ConfigurationSpace,
    ) -> float:
        incumbent = self._configure_wtih_smac(
            portfolio,
            training_instances,
            configuration_space,
        )
        portfolio.update_config(incumbent)
        cost = self._validate(portfolio, training_instances)
        return cost

    def _configure_wtih_smac(
        self,
        portfolio: Portfolio,
        training_instances: InstanceSet,
        configuration_space: ConfigurationSpace,
    ) -> Configuration:
        smac = self._get_smac_algorithm_configuration_facade(configuration_space)
        configuration_time = np.ones(shape=(portfolio.size,)) * self.t_c
        logger.debug(f"SMAC configuration, time: {configuration_time}")
        iteration = 1
        while (configuration_time > 0).any():
            trial_info = smac.ask()
            portfolio.update_config(trial_info.config)
            cost = portfolio.evaluate(
                training_instances,
                configuration_time,
                "configuration",
                calculate_instance_features=self.CALCULATE_INSTANCE_FEATURES,
            )
            logger.debug(
                f"SMAC iteration {iteration}, cost: {cost:.2f}, configuration time: {configuration_time}"
            )
            trial_value = TrialValue(cost=cost)
            smac.tell(trial_info, trial_value)
            iteration += 1
        incumbent = smac.intensifier.get_incumbent()
        return incumbent

    def _get_smac_algorithm_configuration_facade(
        self,
        configuration_space: ConfigurationSpace,
    ):
        self.__set_temp_dir()
        scenario = Scenario(
            configspace=configuration_space,
            output_directory=self.__temp_dir_path,
            use_default_config=False,
            deterministic=True,
            seed=-1,
        )
        intensifier = AlgorithmConfigurationFacade.get_intensifier(
            scenario,
            max_config_calls=1,
        )
        smac = AlgorithmConfigurationFacade(
            scenario,
            lambda seed: None,
            overwrite=True,
            logging_level=logging.CRITICAL,
            intensifier=intensifier,
        )
        return smac

    def _validate(
        self,
        portfolio: Portfolio,
        training_instances: InstanceSet,
    ) -> float:
        validation_time = np.ones(shape=(portfolio.size,)) * self.t_v
        logger.debug(f"Validation, time: {validation_time}")
        cost = portfolio.evaluate(training_instances, validation_time, "validation")
        logger.debug(
            f"Validation cost: {cost:.2f}, remaining validation time: {validation_time}"
        )
        return cost

    def __set_temp_dir(self):
        if hasattr(self, "__temp_dir") and self.__temp_dir is not None:
            self.__cleanup_temp_dir()
        self.__temp_dir = tempfile.TemporaryDirectory(dir=TEMP_DIR)
        self.__temp_dir_path = Path(self.__temp_dir.name)

    def __cleanup_temp_dir(self):
        if self.__temp_dir is not None:
            self.__temp_dir.cleanup()
            self.__temp_dir = None
            self.__temp_dir_path = None
