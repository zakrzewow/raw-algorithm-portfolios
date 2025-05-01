import copy
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from src.database import DB
from src.instance.Instance import Instance
from src.log import logger
from src.surrogate.wrapper import BaseWrapper
from src.utils import hash_str


class Solver(ABC):
    CONFIGURATION_SPACE: ConfigurationSpace

    def __init__(self, config: Configuration = None):
        if config is None:
            config = self.CONFIGURATION_SPACE.sample_configuration()
        self.config = config

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        str_ = ",".join([f"{x:.2f}" for x in self.config.get_array()])
        return hash_str(str_)

    def __repr__(self):
        id_ = self.id()
        str_ = f"Solver(id={id_})"
        return str_

    def copy(self) -> "Solver":
        return copy.deepcopy(self)

    def id(self):
        return str(hash(self))

    def log(self):
        logger.debug(self.__repr__())

    @classmethod
    def from_db(cls, id_: str, db: DB = None) -> "Solver":
        if db is None:
            db = DB()
        dict_ = db.select_id(DB.SCHEMA.SOLVERS, id_)
        vector = list(dict_.values())[1:]
        config = Configuration(cls.CONFIGURATION_SPACE, vector=vector)
        solver = cls(config)
        return solver

    def to_db(self):
        DB().insert(DB.SCHEMA.SOLVERS, self.id(), self.to_dict())
        pass

    def to_dict(self) -> dict:
        dict_ = dict(zip(self.config.keys(), self.config.get_array()))
        return dict_

    def get_array(self) -> np.array:
        return self.config.get_array()

    class Result:
        def __init__(
            self,
            prefix: str,
            solver: "Solver",
            instance: Instance,
            cost: float,
            time: float,
            cached: bool = False,
            surrogate: bool = False,
            error: bool = False,
        ):
            self.prefix = prefix
            self.solver = solver
            self.instance = instance
            self.cost = cost
            self.time = time
            self.cut_off_cost = self.instance.cut_off_cost
            self.cut_off_time = self.instance.cut_off_time
            self.cached = cached
            self.surrogate = surrogate
            self.error = error

        def __repr__(self):
            str_ = (
                f"Solver.Result("
                f"prefix={self.prefix}, "
                f"solver={self.solver}, "
                f"instance={self.instance}, "
                f"cost={self.cost:.2f}, "
                f"time={self.time:.2f}, "
                f"cut_off_cost={self.cut_off_cost:.2f}, "
                f"cut_off_time={self.cut_off_time:.2f}, "
                f"cached={self.cached}, "
                f"surrogate={self.surrogate}, "
                f"error={self.error})"
            )
            return str_

        def evaluation_id(self) -> str:
            return f"{self.solver.id()}_{self.instance.id()}"

        def result_id(self) -> str:
            return f"{self.prefix}_{self.evaluation_id()}"

        def log(self):
            logger.debug(self.__repr__())

        @classmethod
        def from_db(
            cls,
            prefix: str,
            solver: "Solver",
            instance: "Instance",
            features_time: float = 0.0,
        ) -> "Solver.Result":
            id_ = f"{solver.id()}_{instance.id()}"
            dict_ = DB().select_id(DB.SCHEMA.EVALUATIONS, id_)
            if dict_:
                result = cls(
                    prefix=prefix,
                    solver=solver,
                    instance=instance,
                    cost=dict_["cost"],
                    time=features_time,
                    cached=True,
                    surrogate=False,
                    error=False,
                )
                return result
            return None

        def to_db(self):
            if not self.cached and not self.surrogate:
                DB().insert(
                    DB.SCHEMA.EVALUATIONS,
                    self.evaluation_id(),
                    self.to_dict_evaluation(),
                )
            DB().insert(
                DB.SCHEMA.RESULTS,
                self.result_id(),
                self.to_dict_result(),
            )

        def to_dict_evaluation(self) -> dict:
            dict_ = {
                "solver_id": self.solver.id(),
                "instance_id": self.instance.id(),
                "cost": self.cost,
            }
            return dict_

        def to_dict_result(self) -> dict:
            dict_ = {
                "prefix": self.prefix,
                "solver_id": self.solver.id(),
                "instance_id": self.instance.id(),
                "cost": self.cost,
                "time": self.time,
                "cut_off_cost": self.instance.cut_off_cost,
                "cut_off_time": self.instance.cut_off_time,
                "cached": self.cached,
                "surrogate": self.surrogate,
                "error": self.error,
            }
            return dict_

        @classmethod
        def predict_with_estimator(
            cls,
            prefix: str,
            solver: "Solver",
            instance: Instance,
            estimator_wrapper: BaseWrapper,
            features_time: float = 0.0,
        ) -> "Solver.Result":
            X = np.concatenate([solver.get_array(), instance.get_array()])
            X = X.reshape(1, -1)
            cut_off = np.array([instance.cut_off_cost])
            cost = float(estimator_wrapper.predict(X, cut_off)[0])
            result = cls(
                prefix=prefix,
                solver=solver,
                instance=instance,
                cost=cost,
                time=features_time,
                cached=False,
                surrogate=True,
                error=False,
            )
            return result

        @classmethod
        def error_instance(
            cls,
            prefix: str,
            solver: "Solver",
            instance: Instance,
        ) -> "Solver.Result":
            result = cls(
                prefix=prefix,
                solver=solver,
                instance=instance,
                cost=instance.cut_off_cost,
                time=instance.cut_off_time,
                cached=False,
                surrogate=False,
                error=True,
            )
            return result

        def as_future(self) -> Future["Solver.Result"]:
            future = Future()
            future.set_result(self)
            future.add_done_callback(self._future_done_callback)
            return future

        @staticmethod
        def _future_done_callback(future: Future["Solver.Result"]):
            result = future.result()
            if not future.cancelled():
                result.log()
                result.to_db()

    def solve(
        self,
        instance: Instance,
        prefix: str,
        calculate_features: bool = False,
        cache: bool = True,
        estimator_wrapper: BaseWrapper = None,
        executor: ProcessPoolExecutor = None,
    ) -> Future:
        logger.debug(f"solve(prefix={prefix}, solver={self}, instance={instance})")
        features_time = 0.0

        # instance features
        if calculate_features and not instance.features_calculated:
            logger.debug(f"calculate_features(instance={instance})")
            result_with_time = instance.calculate_features(executor=None)
            features_time += result_with_time.time

        # saving to database
        instance.to_db()
        self.to_db()

        # caching
        if cache:
            result = self.Result.from_db(prefix, self, instance, features_time)
            if result is not None:
                return result.as_future()

        # surrogate estimation
        if estimator_wrapper is not None:
            result = self.Result.predict_with_estimator(
                prefix,
                self,
                instance,
                estimator_wrapper,
                features_time,
            )
            return result.as_future()

        # non-paralell
        if executor is None:
            result = self._solve(prefix, self, instance, features_time)
            return result.as_future()

        # paralell
        future = executor.submit(self._solve, prefix, self, instance, features_time)
        future.add_done_callback(self.Result._future_done_callback)
        return future

    @classmethod
    @abstractmethod
    def _solve(
        cls,
        prefix: str,
        solver: "Solver",
        instance: Instance,
        features_time: float = 0.0,
    ) -> "Solver.Result":
        pass
