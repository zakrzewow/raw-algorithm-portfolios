import numpy as np

from src.constant import DATA_DIR
from src.experiment.GlobalExperiment import GlobalExperiment
from src.instance.TSP_Instance import TSP_InstanceSet
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=30,
        seed=0,
    )
    t_c = int(7.5 * 3600 / 10)
    t_v = int(0.5 * 3600)
    K = 4
    n = 4
    solver_class = TSP_LKH_Solver

    experiment = GlobalExperiment(
        t_c=t_c,
        t_v=t_v,
        K=K,
        n=n,
        solver_class=solver_class,
    )

    best_portfolio = experiment.construct_portfolio(train_instances)

    remaining_time = np.ones(shape=(K,)) * np.inf
    best_portfolio.evaluate(test_instances, remaining_time, "test")
