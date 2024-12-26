from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.instance.InstanceList import InstanceList
from src.instance.TSP_Instance import TSP_train_test_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=5,
    )
    test_instances = InstanceList.from_iterable(test_instances[:95])
    portfolio = Portfolio.from_solver_class(TSP_LKH_Solver, size=1)

    aac = AAC(
        portfolio=portfolio,
        instance_list=train_instances,
        prefix="config",
        t_c=7200,
        calculate_features=True,
    )

    for _ in aac.configure():
        pass

    portfolio.evaluate(
        InstanceList.from_iterable(test_instances),
        prefix="test",
        calculate_features=False,
        cache=False,
    )
