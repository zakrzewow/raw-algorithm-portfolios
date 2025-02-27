from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.instance.InstanceList import InstanceList
from src.instance.TSP_Instance import TSP_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    N = 250

    test_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TEST_600" / "index.json",
        max_cost=53.1,
        max_time=5.31,
    )
    number_of_instances = N // 5

    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN_200" / "index.json",
        max_cost=4.7,
        max_time=0.47,
    )
    train_instances_200 = InstanceList()
    for i in range(5):
        train_instances_200.extend(instances[i * 200 : i * 200 + number_of_instances])

    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN_400" / "index.json",
        max_cost=21.8,
        max_time=2.18,
    )
    train_instances_400 = InstanceList()
    for i in range(5):
        train_instances_400.extend(instances[i * 200 : i * 200 + number_of_instances])

    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN_600" / "index.json",
        max_cost=53.1,
        max_time=5.31,
    )
    train_instances_600 = InstanceList()
    for i in range(5):
        train_instances_600.extend(instances[i * 200 : i * 200 + number_of_instances])

    portfolio = Portfolio.from_solver_class(TSP_LKH_Solver, size=2)

    aac = AAC(
        portfolio=portfolio,
        instance_list=train_instances_200,
        prefix="config",
        max_iter=75,
        calculate_features=False,
        estimator=None,
    )
    for _ in aac.configure_iter():
        if aac.iter == 25:
            aac.update(instance_list=train_instances_400)
        elif aac.iter == 50:
            aac.update(instance_list=train_instances_600)

    portfolio = aac._portfolio

    for i in range(100):
        portfolio.evaluate(
            test_instances,
            prefix=f"test{i}",
            calculate_features=False,
            cache=False,
        )
