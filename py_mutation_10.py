from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.database import DB
from src.database.queries import get_number_of_no_timeouts
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

    aac = AAC(
        portfolio=portfolio,
        instance_list=train_instances,
        prefix="config",
        t_c=7200,
        calculate_features=True,
        estimator=None,
    )

    db = DB()

    for _ in aac.configure_iter():
        for idx, instance in enumerate(train_instances):

            number_of_no_timeouts = get_number_of_no_timeouts(
                db, instance.id(), TSP_LKH_Solver.MAX_COST
            )
            if number_of_no_timeouts >= 10:
                new_instance, time = instance.mutate()
                train_instances[idx] = new_instance
                aac._update_configuration_time(time)
            aac.update(instance_list=train_instances)
        train_instances.log()

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
