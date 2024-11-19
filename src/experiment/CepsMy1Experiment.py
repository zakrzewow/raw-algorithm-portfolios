import random
from typing import Type

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from smac.model.random_forest import RandomForest
from smac.runhistory.dataclasses import TrialValue

from src.database import db_connect, db_fetch_result
from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class CepsMy1Experiment(Experiment):
    NAME = "CEPS_MY_1"
    CALCULATE_INSTANCE_FEATURES = True

    def __init__(
        self,
        t_c: int,
        t_v: int,
        t_ini: int,
        t_i: int,
        K: int,
        n: int,
        max_iter: int,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        super().__init__(t_c, t_v, K, n, solver_class, instance_class)
        self.t_ini = t_ini
        self.t_i = t_i
        self.max_iter = max_iter

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        portfolio = self._initialization(train_instances)
        for phase in range(self.max_iter):
            logger.info(f"Phase {phase + 1}/{self.max_iter} -- solvers")
            portfolio.log()
            best_portfolio = None
            best_cost = np.inf

            for _ in range(self.n):
                i = random.choice(range(portfolio.size))
                logger.info(f"Attempt {_ + 1}/{self.n} -- improving solver {i}")
                temp_portfolio = portfolio.copy()
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
                    best_portfolio = temp_portfolio

            portfolio = best_portfolio
            best_portfolio = None
            best_cost = np.inf

            for _ in range(self.n):
                i = random.choice(range(portfolio.size))
                logger.info(f"Attempt {_ + 1}/{self.n} -- EPM improving solver {i}")
                temp_portfolio = portfolio.copy()
                temp_portfolio[i] = self.solver_class()
                configuration_space = portfolio.get_configuration_space(i=i)
                config = {f"{i}__{k}": v for k, v in temp_portfolio[i].config.items()}
                config = Configuration(configuration_space, values=config)
                incumbent = self._configure_wtih_smac_and_surrogate_model(
                    temp_portfolio,
                    train_instances,
                    configuration_space,
                    config,
                    n_trials=50,
                )
                temp_portfolio.update_config(incumbent)
                cost = self._validate(temp_portfolio, train_instances)
                logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
                if cost < best_cost:
                    best_cost = cost
                    best_portfolio = temp_portfolio

            portfolio = best_portfolio
            if phase == self.max_iter - 1:
                break

            logger.info(f"Phase {phase + 1}/{self.max_iter} -- instances")
            portfolio.log()
            t = {instance.__hash__(): instance for instance in train_instances}
            tprim = {instance.__hash__(): instance for instance in train_instances}
            costs = {}
            for k, instance in tprim.items():
                costs[k] = portfolio.evaluate(
                    InstanceSet.from_instance_list([instance]),
                    np.ones(shape=(self.K,)) * np.inf,
                    "pre_mutation",
                    calculate_instance_features=self.CALCULATE_INSTANCE_FEATURES,
                )
            mutation_time = np.ones(shape=(self.K,)) * self.t_i
            while (mutation_time > 0).any():
                instance = random.choice(train_instances)
                logger.info(f"Mutating {instance.__hash__()}, time = {mutation_time}")
                instance, time = instance.mutate()
                mutation_time -= time
                cost = portfolio.evaluate(
                    InstanceSet.from_instance_list([instance]),
                    mutation_time,
                    "mutation",
                    calculate_instance_features=self.CALCULATE_INSTANCE_FEATURES,
                )
                lower_cost_instances = [k for k, v in costs.items() if v < cost]
                if len(lower_cost_instances) > 0:
                    k = random.choice(lower_cost_instances)
                    logger.debug(
                        f"Replacing {k} (cost={costs[k]}) with {instance.__hash__()} (cost={cost})"
                    )
                    del costs[k]
                    del tprim[k]
                    costs[instance.__hash__()] = cost
                    tprim[instance.__hash__()] = instance
            t = {**t, **tprim}
            train_instances = InstanceSet.from_instance_list(list(t.values()))
            train_instances.log()

        return portfolio

    def _initialization(self, train_instances: InstanceSet) -> Portfolio:
        solvers = []
        logger.info("Initialization")

        num_of_configs = (
            self.t_ini
            // sum([i * train_instances.size * 10 for i in range(1, self.K + 1)])
            + 1
        )
        configuration_list = [
            self.solver_class.CONFIGURATION_SPACE.sample_configuration()
            for _ in range(num_of_configs)
        ]

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
                    calculate_instance_features=self.CALCULATE_INSTANCE_FEATURES,
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

    def _configure_wtih_smac_and_surrogate_model(
        self,
        portfolio: Portfolio,
        train_instances: InstanceSet,
        configuration_space: ConfigurationSpace,
        initial_configuration: Configuration = None,
        n_trials: int = 50,
    ) -> Configuration:
        epm = self._get_epm()
        conn = db_connect()
        smac = self._get_smac_algorithm_configuration_facade(
            configuration_space,
            initial_configuration,
        )
        logger.debug(f"SMAC configuration with surrogate model")
        for _ in range(n_trials):
            trial_info = smac.ask()
            portfolio.update_config(trial_info.config)
            shape = (train_instances.size, portfolio.size)
            max_cost = np.array([s.MAX_COST for s in portfolio])
            costs = np.ones(shape=shape) * max_cost
            for i in range(train_instances.size):
                for j in range(portfolio.size):
                    cached_result = db_fetch_result(
                        conn, train_instances[i], portfolio[j]
                    )
                    if cached_result is not None:
                        cost, time = cached_result
                        costs[i, j] = cost
                    else:
                        solver_array = portfolio[j].config.get_array().reshape(1, -1)
                        instance_id = train_instances[i].__hash__()
                        query = """
                        select 
                            instances.*
                        from instances
                        where instances.id = ?
                        """
                        instance_array = (
                            pd.read_sql_query(query, conn, params=(instance_id,))
                            .drop(columns="id")
                            .to_numpy()
                        )
                        X = np.concatenate([solver_array, instance_array], axis=1)
                        costs[i, j] = epm.predict(X)[0][0][0]
                        logger.debug(f"({i}, {j}) EPM cost: {costs[i, j]}")
            cost = costs.min(axis=1).mean()
            logger.debug(f"SMAC iteration {_ + 1}, cost: {cost:.2f}")
            trial_value = TrialValue(cost=cost)
            smac.tell(trial_info, trial_value)
        incumbent = smac.intensifier.get_incumbent()
        return incumbent

    def _get_epm(self):
        conn = db_connect()
        query = """
        select 
            instances.*
        from results
        join instances on results.instance_id = instances.id
        """

        instance_features = pd.read_sql_query(query, conn).drop(columns="id")
        instance_features = instance_features.T.to_dict(orient="list")

        query = """
        select 
            results.cost,
            solvers_f64.*,
            instances.*
        from results
        join instances on results.instance_id = instances.id
        join solvers_f64 on results.solver_id = solvers_f64.id
        """

        df = pd.read_sql_query(query, conn).drop(columns="id")
        conn.close()

        y = df["cost"].to_numpy()
        X = df.drop(columns="cost").to_numpy()

        epm = RandomForest(
            configspace=self.solver_class.CONFIGURATION_SPACE,
            seed=0,
            instance_features=instance_features,
        )
        epm = epm.train(X, y)
        return epm
