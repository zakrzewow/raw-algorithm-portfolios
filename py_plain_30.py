from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.instance.InstanceList import InstanceList
from src.instance.TSP_Instance import TSP_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    N = 30

    test_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TEST" / "index.json"
    )
    instances = TSP_from_index_file(filepath=DATA_DIR / "TSP" / "TRAIN" / "index.json")
    number_of_instances = N // 5
    train_instances = InstanceList()
    for i in range(5):
        train_instances.extend(instances[i * 200 : i * 200 + number_of_instances])

    portfolio = Portfolio.from_solver_class(TSP_LKH_Solver, size=2)

    aac = AAC(
        portfolio=portfolio,
        instance_list=train_instances,
        prefix="config",
        max_iter=75,
        calculate_features=True,
        estimator=None,
    )
    aac.configure()

    for i in range(100):
        portfolio.evaluate(
            test_instances,
            prefix=f"test{i}",
            calculate_features=False,
            cache=False,
        )
