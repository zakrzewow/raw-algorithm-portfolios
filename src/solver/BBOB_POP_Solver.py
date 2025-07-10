import nevergrad as ng
from ConfigSpace import Configuration

from src.configuration_space.POP import CONFIGURATION_SPACE
from src.instance import BBOB_Instance
from src.solver.Solver import Solver
from src.utils import Timer


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

        optimizer = optimizer_class(parametrization=problem.dimension, budget=1e6)

        with Timer() as timer:
            try:
                _ = optimizer.minimize(
                    problem,
                    max_time=instance.cut_off_time,
                )
            except Exception:
                error = True
                time = instance.cut_off_time
                cost = instance.cut_off_cost

        error = False
        if not problem.final_target_hit:
            time = instance.cut_off_time
            cost = instance.cut_off_cost
        else:
            time = timer.elapsed_time
            cost = time if time < instance.cut_off_time else instance.cut_off_cost

        time += features_time
        return Solver.Result(prefix, solver, instance, cost, time, error=error)
