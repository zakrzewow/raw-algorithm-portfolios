from src.constant import DATA_DIR
from src.instance.InstanceList import InstanceList
from src.instance.TSP_Instance import TSP_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    N = 100

    all_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN" / "index.json",
        max_cost=3000.0,
        max_time=300.0,
    )

    number_of_instances = N // 5
    instances = InstanceList()
    for i in range(5):
        instances.extend(all_instances[i * 200 : i * 200 + number_of_instances])

    portfolio = Portfolio.from_solver_class(TSP_LKH_Solver, size=500)
    portfolio.evaluate(
        instance_list=instances,
        prefix="dataset",
        calculate_features=True,
        cache=True,
    )
