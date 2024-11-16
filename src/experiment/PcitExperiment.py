import random
from typing import Dict, Type

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from smac.model.random_forest import RandomForest

from src.database import db_connect
from src.experiment import Experiment
from src.instance import Instance, InstanceSet
from src.log import logger
from src.portfolio import Portfolio
from src.solver import Solver


class _Clustering:
    def __init__(
        self,
        portfolio: Portfolio,
        instances: InstanceSet,
    ):
        self.portfolio = portfolio
        self.instances = instances
        self._cluster_assignments = self._get_random_cluster_assignments()

    def _get_random_cluster_assignments(self) -> Dict[str, int]:
        cluster_assignments = np.random.choice(self.portfolio.size, self.instances.size)
        while True:
            counts = np.bincount(cluster_assignments, minlength=self.portfolio.size)
            if max(counts) - min(counts) <= 1:
                break
            cluster_assignments = np.random.choice(
                self.portfolio.size, self.instances.size
            )

        cluster_assignments = {
            self.instances[i].__hash__(): j for i, j in enumerate(cluster_assignments)
        }
        return cluster_assignments

    def get_instances_for_solver(self, i: int):
        instance_list = [
            self.instances[j]
            for j in range(self.instances.size)
            if self._cluster_assignments[self.instances[j].__hash__()] == i
        ]
        return InstanceSet.from_instance_list(instance_list)

    def get_solver_id(self, instance_id):
        i = self._cluster_assignments[instance_id]
        return hash(self.portfolio[i])

    def assign_instance(self, instance_id, solver_id):
        solver_i = [
            i
            for i in range(self.portfolio.size)
            if hash(self.portfolio[i]) == solver_id
        ][0]
        self._cluster_assignments[instance_id] = solver_i

    @property
    def instance_id_solver_id(self):
        return {
            k: hash(self.portfolio[v]) for k, v in self._cluster_assignments.items()
        }

    @property
    def solver_id_instance_count(self):
        counts = {hash(self.portfolio[i]): 0 for i in range(self.portfolio.size)}
        for i in self._cluster_assignments.values():
            counts[hash(self.portfolio[i])] += 1
        return counts


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
        self.lower_size = self.upper_size = None

    def construct_portfolio(self, train_instances: InstanceSet) -> Portfolio:
        best_portfolio = None
        best_cost = np.inf
        T_C = self.t_c
        self.lower_size = np.ceil(0.8 * train_instances.size / self.K)
        self.upper_size = np.ceil(1.2 * train_instances.size / self.K)

        for _ in range(self.n):
            logger.info(f"Attempt {_ + 1}/{self.n}")
            portfolio = Portfolio.from_solver_class(self.solver_class, self.K)
            clustering = _Clustering(portfolio, train_instances)
            logger.info(f"Clustering: {clustering._cluster_assignments}")

            for phase in range(self.max_iter):
                logger.info(f"Phase {phase + 1}/{self.max_iter}")
                portfolio.log()
                if phase == self.max_iter - 1:
                    self.t_c = T_C / 2
                else:
                    self.t_c = T_C / (2 * (self.max_iter - 1))

                for i in range(self.K):
                    logger.info(f"Solver {i + 1}/{self.K}")
                    solver = portfolio[i]
                    temp_portfolio = Portfolio.from_solver_list([solver])
                    temp_instances = clustering.get_instances_for_solver(i)
                    configuration_space = temp_portfolio.get_configuration_space()
                    config = {f"0__{k}": v for k, v in solver.config.items()}
                    config = Configuration(configuration_space, values=config)
                    incumbent = self._configure_wtih_smac(
                        temp_portfolio,
                        temp_instances,
                        configuration_space,
                        config,
                    )
                    temp_portfolio.update_config(incumbent)
                    portfolio[i] = temp_portfolio[0]
                self.instance_transfer(portfolio, clustering)

            cost = self._validate(portfolio, train_instances)
            logger.info(f"Attempt {_ + 1}/{self.n}: cost = {cost:.2f}")
            if cost < best_cost:
                logger.info(f"New best portfolio found!")
                best_cost = cost
                best_portfolio = portfolio
        return best_portfolio

    def instance_transfer(self, portfolio: Portfolio, clustering: _Clustering):
        logger.debug("Instance transfer")
        epm = self._get_epm()
        t = self._get_instances_to_transfer(clustering)
        logger.debug(f"Instances to transfer: {t}")

        while True:
            t_done = {}
            t_remain = {}
            while len(t) > 0:
                instance_id = random.choice(list(t.keys()))
                currenct_solver_id = clustering.get_solver_id(instance_id)
                logger.debug(
                    f"Instance {instance_id} is currently with solver {currenct_solver_id}"
                )
                del t[instance_id]

                expected_performance = self._get_expected_performance(
                    instance_id,
                    portfolio,
                    epm,
                )
                solver_id_instance_count = clustering.solver_id_instance_count
                for solver_id, performance in expected_performance.items():
                    logger.debug(
                        f"Instance {instance_id} comparing current {currenct_solver_id}={expected_performance[currenct_solver_id]} with {solver_id}={performance}"
                    )
                    if (
                        performance < expected_performance[currenct_solver_id]
                        and solver_id_instance_count[solver_id] < self.upper_size
                        and solver_id_instance_count[currenct_solver_id]
                        > self.lower_size
                    ):
                        logger.debug(
                            f"Instance {instance_id} will be transferred to solver {solver_id}"
                        )
                        clustering.assign_instance(instance_id, solver_id)
                        t_done[instance_id] = solver_id
                        break
                if instance_id not in t_done:
                    logger.debug(
                        f"Instance {instance_id} will remain with solver {currenct_solver_id}"
                    )
                    t_remain[instance_id] = currenct_solver_id
            t = t_remain
            if len(t_done) == 0 or len(t) == 0:
                break

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

    def _get_instances_to_transfer(self, clustering: _Clustering):
        conn = db_connect()
        incumbent_performance = {}

        for instance_id, solver_id in clustering.instance_id_solver_id.items():
            query = (
                "SELECT MIN(cost) FROM results WHERE instance_id = ? AND solver_id = ?"
            )
            cursor = conn.cursor()
            cursor.execute(query, (instance_id, solver_id))
            result = cursor.fetchone()
            incumbent_performance[instance_id] = result[0]
            cursor.close()
        conn.close()
        median = np.median(list(incumbent_performance.values()))
        t = {k: v for k, v in incumbent_performance.items() if v >= median}
        return t

    def _get_expected_performance(self, instance_id, portfolio, epm):
        conn = db_connect()
        expected_performance = {}

        for solver in portfolio:
            solver_id = hash(solver)
            query = """
            select 
                solvers_f64.*,
                instances.*
            from instances
            join solvers_f64 on 1 = 1
            where instances.id = ? and solvers_f64.id = ?
            """
            X = (
                pd.read_sql_query(query, conn, params=(instance_id, solver_id))
                .drop(columns="id")
                .to_numpy()
            )
            expected_performance[solver_id] = epm.predict(X)[0][0][0]
        conn.close()
        expected_performance = dict(
            sorted(expected_performance.items(), key=lambda item: item[1])
        )
        return expected_performance
