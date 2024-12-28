import numpy as np

from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_train_test_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=5,
    )
    test_instances = test_instances[:95]
    portfolio = Portfolio.from_solver_class(TSP_LKH_Solver, size=1)

    def get_bin_index(bin_edges, x):
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                return i + 1
        return i + 1

    actual_train_instances = train_instances[:1]

    aac = AAC(
        portfolio=portfolio,
        instance_list=actual_train_instances,
        prefix="config",
        t_c=7200,
        calculate_features=True,
        estimator=None,
    )

    bin_edges = np.full(
        train_instances.size, 1 / sum(range(1, train_instances.size + 1))
    )
    bin_edges = np.cumsum(bin_edges)
    bin_edges = np.insert(bin_edges, 0, 0)
    bin_edges = np.cumsum(bin_edges)

    for _ in aac.configure_iter():
        progress = aac.get_progress()
        new_size = get_bin_index(bin_edges, progress)
        if new_size != actual_train_instances.size:
            actual_train_instances = train_instances[:new_size]
            aac.update(instance_list=actual_train_instances)
            actual_train_instances.log()

    portfolio.evaluate(
        test_instances,
        prefix="test1",
        calculate_features=False,
        cache=False,
    )
    portfolio.evaluate(
        test_instances,
        prefix="test2",
        calculate_features=False,
        cache=False,
    )
    portfolio.evaluate(
        test_instances,
        prefix="test3",
        calculate_features=False,
        cache=False,
    )
