import os

from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    generator = os.environ.get("GENERATOR").strip()
    generator_to_i = {
        "cluster_netgen": 0,
        "compression": 200,
        "expansion": 400,
        "explosion": 600,
        "grid": 800,
    }
    i = generator_to_i[generator]

    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN" / "index.json",
        cut_off_cost=3000.0,
        cut_off_time=300.0,
    )
    instances = instances[i : i + 20]

    portfolio = Portfolio.from_solver_class(TSP_LKH_Solver, size=1000)
    portfolio.evaluate(
        instance_list=instances,
        prefix="dataset",
        calculate_features=False,
        cache=True,
    )
