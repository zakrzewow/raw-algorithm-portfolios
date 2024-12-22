import copy
from abc import ABC, abstractmethod

from ConfigSpace import Configuration, ConfigurationSpace

from src.database import DB
from src.instance.Instance import Instance
from src.log import logger
from src.utils import ResultWithTime, hash_str


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

    def to_db(self):
        DB().insert(DB.SCHEMA.SOLVERS, self.id(), self.to_dict())
        pass

    def to_dict(self) -> dict:
        dict_ = dict(zip(self.config.keys(), self.config.get_array()))
        return dict_

    @classmethod
    def from_db(cls, id_: str) -> "Solver":
        dict_ = DB().select_id(DB.SCHEMA.SOLVERS, id_)
        vector = list(dict_.values())[1:]
        config = Configuration(cls.CONFIGURATION_SPACE, vector=vector)
        solver = cls(config)
        return solver

    @abstractmethod
    def solve(self, instance: Instance) -> ResultWithTime:
        pass
