from ConfigSpace import Configuration

from src.configuration_space.POP import CONFIGURATION_SPACE
from src.instance import BBOB_Instance
from src.solver.Solver import Solver


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
        pass

        # try:
        #     result = subprocess.run(
        #         [LKH_PATH, config_filepath],
        #         capture_output=True,
        #         text=True,
        #         stdin=subprocess.DEVNULL,
        #         timeout=instance.cut_off_time + 5,
        #     )
        #     time = solver._parse_result(result, instance)
        #     cost = time if time < instance.cut_off_time else instance.cut_off_cost
        #     error = False
        # except subprocess.TimeoutExpired:
        #     time = instance.cut_off_time
        #     cost = instance.cut_off_cost
        #     error = True
        # time += features_time
        # solver._remove_config_file(config_filepath)
        # return Solver.Result(prefix, solver, instance, cost, time, error=error)
