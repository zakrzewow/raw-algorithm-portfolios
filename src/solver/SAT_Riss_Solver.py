import subprocess

from ConfigSpace import Configuration

from src.configuration_space.Riss import CONFIGURATION_SPACE
from src.constant import RISS_PATH
from src.instance import SAT_Instance
from src.solver.Solver import Solver


class SAT_Riss_Solver(Solver):
    CONFIGURATION_SPACE = CONFIGURATION_SPACE

    def __init__(self, config: Configuration = None):
        super().__init__(config)

    @classmethod
    def _solve(
        cls,
        prefix: str,
        solver: "SAT_Riss_Solver",
        instance: SAT_Instance,
        features_time: float = 0.0,
    ) -> Solver.Result:
        params = solver._get_params()
        try:
            result = subprocess.run(
                [RISS_PATH, instance.filepath, *params],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=instance.max_time + 5,
            )
            time = solver._parse_result(result, instance)
            cost = time if time < instance.max_time else instance.max_cost
            error = False
        except subprocess.TimeoutExpired:
            time = instance.max_time
            cost = instance.max_cost
            error = True
        except Exception as e:
            print(f"Error running {solver=} on {instance=}: {e}")
            time = instance.max_time
            cost = instance.max_cost
            error = True
        time += features_time
        return Solver.Result(prefix, solver, instance, cost, time, error=error)

    def _get_params(self) -> list[str]:
        params = []
        for k, v in self.config.items():
            if isinstance(v, str):
                if v == "on":
                    params.append(f"-{k}")
                elif v == "off":
                    params.append(f"-no-{k}")
                else:
                    print("Unknown string value:", v)
            elif isinstance(v, int):
                params.append(f"-{k}={v}")
            elif isinstance(v, float):
                params.append(f"-{k}={v:.4f}")
        return params

    def _parse_result(
        self,
        result: subprocess.CompletedProcess,
        instance: SAT_Instance,
    ) -> float:
        time = None
        for line in result.stdout.splitlines():
            if "c CPU time" in line and line.strip().endswith("s"):
                time_str = line.split(":")[-1].strip().replace("s", "").strip()
                time = float(time_str)
            if "s UNKNOWN" in line:
                raise Exception(f"Solver returned {line.strip()}")
        if time is None:
            raise Exception("CPU time not found")
        return min(time, instance.max_time)
