import os

from src.instance.BBOB_Instance import BBOB_Instance
from src.instance.InstanceList import InstanceList
from src.solver.BBOB_POP_Solver import BBOB_POP_Solver
from src.solver.Portfolio import Portfolio

if __name__ == "__main__":
    function_index = int(os.environ.get("FUNCTION_INDEX").strip())

    dimension_list = [2, 3, 5, 10, 20]
    instances = InstanceList()

    for dimension in dimension_list:
        instance = BBOB_Instance(
            function_index=function_index,
            dimension=dimension,
            instance_index=1,
            cut_off_cost=3000.0,
            cut_off_time=300.0,
        )
        instances.append(instance)

    portfolio = Portfolio.from_solver_class(BBOB_POP_Solver, size=1000)
    portfolio.evaluate(
        instance_list=instances,
        prefix="dataset",
        calculate_features=False,
        cache=True,
    )
