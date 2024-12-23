import copy
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor

from ConfigSpace import Configuration, ConfigurationSpace

from src.database import DB
from src.instance.Instance import Instance
from src.log import logger
from src.utils import hash_str


class Solver(ABC):
    CONFIGURATION_SPACE: ConfigurationSpace
    MAX_COST = 0.0
    MAX_TIME = 0.0

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
    def from_db(cls, id_: str) -> "Solver":
        dict_ = DB().select_id(DB.SCHEMA.SOLVERS, id_)
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

    class Result:
        def __init__(
            self,
            solver: "Solver",
            instance: Instance,
            cost: float,
            time: float,
            cached: bool = False,
        ):
            self.solver = solver
            self.instance = instance
            self.cost = cost
            self.time = time
            self.cached = cached

        def __repr__(self):
            str_ = f"Solver.Result(solver={self.solver}, instance={self.instance}, cost={self.cost:.2f}, time={self.time:.2f}, cached={self.cached})"
            return str_

        def id(self) -> str:
            return f"{self.solver.id()}_{self.instance.id()}"

        def log(self):
            logger.debug(self.__repr__())

        @classmethod
        def from_db(cls, solver: "Solver", instance: "Instance") -> "Solver.Result":
            id_ = f"{solver.id()}_{instance.id()}"
            dict_ = DB().select_id(DB.SCHEMA.RESULTS, id_)
            if dict_:
                result = cls(
                    solver=solver,
                    instance=instance,
                    cost=dict_["cost"],
                    time=0,
                    cached=True,
                )
                return result
            return None

        def to_db(self):
            DB().insert(DB.SCHEMA.RESULTS, self.id(), self.to_dict())
            pass

        def to_dict(self) -> dict:
            dict_ = {
                "solver_id": self.solver.id(),
                "instance_id": self.instance.id(),
                "cost": self.cost,
            }
            return dict_

        def as_future(self) -> Future["Solver.Result"]:
            future = Future()
            future.set_result(self)
            future.add_done_callback(self._future_done_callback)
            return future

        @staticmethod
        def _future_done_callback(future: Future["Solver.Result"]):
            result = future.result()
            result.log()
            result.to_db()

    def solve(
        self,
        instance: Instance,
        executor: ProcessPoolExecutor = None,
        cache: bool = True,
        calculate_features: bool = False,
    ) -> Future:
        logger.debug(f"{self} {instance} solving...")
        time = 0.0

        # instance features
        if calculate_features and not instance.features_calculated:
            logger.debug(f"{instance} calculating features...")
            result_with_time = instance.calculate_features()
            time += result_with_time.time
            logger.debug(f"{instance} features calculated {result_with_time}")

        # saving to database
        instance.to_db()
        self.to_db()

        # caching
        if cache and (result := self.Result.from_db(self, instance)) is not None:
            return result.as_future()

        # non-paralell
        if executor is None:
            result = self._solve(self, instance)
            return result.as_future()

        # paralell
        future = executor.submit(self._solve, self, instance)
        future.add_done_callback(self.Result._future_done_callback)
        return future

    @classmethod
    @abstractmethod
    def _solve(cls, solver: "Solver", instance: Instance) -> "Solver.Result":
        pass
