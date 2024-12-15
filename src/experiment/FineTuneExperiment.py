import random
import sqlite3
import time
from typing import Type

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from smac.runhistory.dataclasses import TrialValue

from src.constant import DATA_DIR, MAIN_DIR
from src.database import db_connect, db_fetch_result, db_init
from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.instance.TSP_Instance import TSP_Instance, TSP_InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver


class FineTuneExperiment(Experiment):
    NAME = "FINE_TUNE"
    CALCULATE_INSTANCE_FEATURES = True

    def __init__(
        self,
        db_name: str,
        solver_class: Type[Solver],
        instance_class: Type[Instance],
    ):
        super().__init__(None, None, None, solver_class, instance_class)
        self.db_name = db_name

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        conn_old = sqlite3.connect(MAIN_DIR / "INSTANCES.db")
        df_instances = pd.read_sql_query("SELECT * FROM instances", conn_old)
        conn_old.close()

        conn_old = sqlite3.connect(MAIN_DIR / self.db_name)
        df_solvers = pd.read_sql_query("SELECT * FROM solvers", conn_old)
        df_solvers_f64 = pd.read_sql_query("SELECT * FROM solvers_f64", conn_old)
        df = pd.read_sql_query(
            "SELECT * FROM results",
            conn_old,
        )

        portfolio = pd.read_sql_query(
            """
        WITH portfolio AS (
            SELECT
                DISTINCT solver_id 
            FROM results WHERE comment LIKE 'test%'
        )
        SELECT solvers.* 
        FROM solvers
        JOIN portfolio ON solvers.id = portfolio.solver_id
        """,
            conn_old,
        )
        conn_old.close()

        portfolio = [
            Configuration(
                TSP_LKH_Solver.CONFIGURATION_SPACE, values=config.drop(["id"]).to_dict()
            )
            for _, config in portfolio.iterrows()
        ]
        portfolio = [TSP_LKH_Solver(config) for config in portfolio]
        portfolio = Portfolio.from_solver_list(portfolio)

        conn = db_connect()
        df_instances.to_sql("instances", conn, if_exists="append", index=False)
        df_solvers.to_sql("solvers", conn, if_exists="append", index=False)
        df_solvers_f64.to_sql("solvers_f64", conn, if_exists="append", index=False)
        df.to_sql("results", conn, if_exists="append", index=False)
        return portfolio

    def fine_tune_all(
        self,
        portfolio: Portfolio,
        instance_set: InstanceSet,
        comment="",
    ):
        epm = self._get_epm()
        times = []
        for instance in instance_set:
            try:
                fime_tune_time, cost = self.fine_tune(epm, portfolio, instance, comment)
                times.append(fime_tune_time)
            except Exception as e:
                logger.error(f"Error in fine-tuning: {e}")
        logger.info(f"Fine-tuning time: {pd.Series(times).describe()}")

    def fine_tune(self, epm, portfolio: Portfolio, instance: Instance, comment=""):
        start_time = time.time()
        rf_classifier, rf_regressor = epm
        portfolio = portfolio.copy()
        conn = db_connect()
        instance_set = InstanceSet.from_instance_list([instance])
        instance_id = instance.__hash__()

        max_cost = np.array([s.MAX_COST for s in portfolio])
        costs = np.ones(shape=portfolio.size) * max_cost
        for i in range(portfolio.size):
            solver_array = portfolio[i].config.get_array().reshape(1, -1)
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
            is_timeout = ~rf_classifier.predict(X)[0]
            if is_timeout:
                costs_pred = self.solver_class.MAX_COST
            if not is_timeout:
                costs_pred = rf_regressor.predict(X)[0]
            costs[i] = costs_pred

        i = costs.argmin()
        logger.info(f"EPM predicted costs: {costs}")
        logger.info(f"EPM improving solver {i + 1}/{portfolio.size}")
        portfolio[i] = self.solver_class()
        configuration_space = portfolio.get_configuration_space(i=i)
        config = {f"{i}__{k}": v for k, v in portfolio[i].config.items()}
        config = Configuration(configuration_space, values=config)
        #
        smac = self._get_smac_algorithm_configuration_facade(
            configuration_space,
            config,
        )
        for _ in range(100):
            trial_info = smac.ask()
            portfolio.update_config(trial_info.config)
            solver_array = portfolio[i].config.get_array().reshape(1, -1)
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
            is_timeout = ~rf_classifier.predict(X)[0]
            if is_timeout:
                cost = self.solver_class.MAX_COST
            if not is_timeout:
                cost = rf_regressor.predict(X)[0]
            trial_value = TrialValue(cost=cost)
            smac.tell(trial_info, trial_value)
        incumbent = smac.intensifier.get_incumbent()
        #
        end_time = time.time()
        elapsed_time = end_time - start_time
        portfolio.update_config(incumbent)
        cost = portfolio.evaluate(instance_set, comment=comment, cache=False)
        return elapsed_time, cost

    def _get_epm(self):
        conn = db_connect()
        query = """
        select 
            results.cost,
            solvers_f64.*,
            instances.*
        from results
        join instances on results.instance_id = instances.id
        join solvers_f64 on results.solver_id = solvers_f64.id
        WHERE comment in ('initialization', 'configuration')
        """

        df = pd.read_sql_query(query, conn).drop(columns="id")
        conn.close()

        y = df["cost"].to_numpy()
        X = df.drop(columns="cost").to_numpy()
        y_timeout = y != self.solver_class.MAX_COST

        rf_classifier = RandomForestClassifier(
            class_weight="balanced",
            max_depth=50,
            max_features=0.4,
            min_samples_leaf=8,
            min_samples_split=20,
            n_estimators=200,
            random_state=0,
            n_jobs=10,
        )
        rf_classifier.fit(X, y_timeout)

        X_score = X[y < self.solver_class.MAX_COST]
        y_score = y[y < self.solver_class.MAX_COST]
        rf_regressor = RandomForestRegressor(
            max_depth=20,
            max_features=0.4,
            min_samples_leaf=8,
            min_samples_split=40,
            n_estimators=200,
            random_state=0,
            n_jobs=10,
        )
        rf_regressor = rf_regressor.fit(X_score, y_score)
        return rf_classifier, rf_regressor
