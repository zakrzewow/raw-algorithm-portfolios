from src.constant import DATA_DIR
from src.experiment.PcitExperiment import PcitExperiment
from src.instance.TSP_Instance import TSP_Instance, TSP_InstanceSet
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=30,
        seed=0,
    )
    t_c = int(2 * 3600)
    K = 4
    n = 10
    max_iter = 4
    solver_class = TSP_LKH_Solver
    instance_class = TSP_Instance

    experiment = PcitExperiment(
        t_c=t_c,
        K=K,
        n=n,
        max_iter=max_iter,
        solver_class=solver_class,
        instance_class=instance_class,
    )

    best_portfolio = experiment.construct_portfolio(train_instances)

    best_portfolio.evaluate(test_instances, comment="test1", cache=False)
    best_portfolio.evaluate(test_instances, comment="test2", cache=False)
    best_portfolio.evaluate(test_instances, comment="test3", cache=False)
