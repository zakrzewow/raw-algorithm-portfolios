import numpy as np

from src.constant import DATA_DIR
from src.experiment.CepsMy1Experiment import CepsMy1Experiment
from src.instance.TSP_Instance import TSP_Instance, TSP_InstanceSet
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=30,
        seed=0,
    )

    t_c = int(1.5 * 3600)
    t_v = int(0.5 * 3600)
    K = 4
    n = 10
    t_ini = int(8 * 3600)
    t_i = int(1 * 3600)
    max_iter = 4
    solver_class = TSP_LKH_Solver
    instance_class = TSP_Instance

    experiment = CepsMy1Experiment(
        t_c=t_c,
        t_v=t_v,
        t_ini=t_ini,
        t_i=t_i,
        K=K,
        n=n,
        max_iter=max_iter,
        solver_class=solver_class,
        instance_class=instance_class,
    )

    best_portfolio = experiment.construct_portfolio(train_instances)

    remaining_time = np.ones(shape=(K,)) * np.inf
    best_portfolio.evaluate(test_instances, remaining_time, "test")
