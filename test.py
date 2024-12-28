from src.aac.AAC import AAC
from src.aac.SurrogateEstimator import Estimator1
from src.constant import DATA_DIR
from src.database import DB
from src.database.queries import get_model_training_data
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
        t_c=5400,
        calculate_features=True,
        estimator=None,
    )

    # for _ in aac.configure():
    #     pass

    last_model_iter = 0

    estimator = None
    db = DB()
    estimator_pct = 0.5
    for _ in aac.configure_iter():
        if aac.get_progress() >= 0.5 and aac.iter - last_model_iter >= 10:
            X, y = get_model_training_data(db)
            estimator = Estimator1(
                max_cost=TSP_LKH_Solver.MAX_COST, estimator_pct=estimator_pct
            )
            estimator.fit(X, y)
            estimator.log()
            last_model_iter = aac.iter
            aac.update(estimator=estimator)

    portfolio.evaluate(
        InstanceList.from_iterable(test_instances),
        prefix="test",
        calculate_features=False,
        cache=False,
    )
