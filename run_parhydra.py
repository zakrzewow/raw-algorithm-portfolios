from src.constant import DATA_DIR
from src.experiment.ParhydraExperiment import ParhydraExperiment
from src.instance.TSP_Instance import TSP_InstanceSet
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=5,
        seed=0,
    )
    t_c = 100
    t_v = 100
    K = 2
    n = 2
    solver_class = TSP_LKH_Solver

    experiment = ParhydraExperiment(
        t_c=t_c,
        t_v=t_v,
        K=K,
        n=n,
        solver_class=solver_class,
    )

    experiment.construct_portfolio(train_instances)
