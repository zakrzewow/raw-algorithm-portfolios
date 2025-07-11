import time
import warnings

import nevergrad as ng
from ConfigSpace import Configuration
from nevergrad.common.errors import NevergradRuntimeWarning

from src.configuration_space.POP import CONFIGURATION_SPACE
from src.instance.BBOB_Instance import BBOB_Instance
from src.solver.Solver import Solver

warnings.filterwarnings("ignore", category=NevergradRuntimeWarning)


class BBOB_POP_Solver(Solver):
    CONFIGURATION_SPACE = CONFIGURATION_SPACE

    def __init__(self, config: Configuration = None):
        super().__init__(config)

    @classmethod
    def _solve(
        cls,
        prefix: str,
        solver: "BBOB_POP_Solver",
        instance: BBOB_Instance,
        features_time: float = 0.0,
    ) -> Solver.Result:
        problem = instance.get_problem()

        kwargs = dict(solver.config)
        algorithm = solver.config["ALGORITHM"]

        def _format_key(key: str, algorithm) -> str:
            key = key[len(algorithm) + 1 :]
            if not (algorithm == "DE" and key in ["F1", "F2"]):
                key = key.lower()
            return key

        kwargs = {
            _format_key(k, algorithm): v
            for k, v in kwargs.items()
            if k.startswith(algorithm)
        }

        if algorithm == "PSO":
            optimizer_class = ng.families.ConfPSO(**kwargs)
        elif algorithm == "DE":
            optimizer_class = ng.families.DifferentialEvolution(**kwargs)
        elif algorithm == "CMA":
            optimizer_class = ng.families.ParametrizedCMA(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        optimizer = optimizer_class(parametrization=problem.dimension, budget=100000)

        start_time = time.time()
        try:
            for _ in range(optimizer.budget):
                x = optimizer.ask()
                value = problem(*x.args, **x.kwargs)
                optimizer.tell(x, value)

                end_time = time.time()
                elapsed_time = end_time - start_time

                if problem.final_target_hit or elapsed_time >= instance.cut_off_time:
                    break
        except Exception:
            error = True
            time_ = instance.cut_off_time
            cost = instance.cut_off_cost

        error = False
        if not problem.final_target_hit:
            time_ = instance.cut_off_time
            cost = instance.cut_off_cost
        else:
            time_ = elapsed_time
            if time_ < instance.cut_off_time:
                cost = time_
            else:
                time_ = instance.cut_off_time
                cost = instance.cut_off_cost
        time_ += features_time
        return Solver.Result(prefix, solver, instance, cost, time_, error=error)
