from src.aac.AAC import AAC
from src.aac.SurrogateEstimator import Estimator1
from src.constant import DATA_DIR
from src.database import DB
from src.database.queries import get_model_training_data
from src.instance.SAT_Instance import SAT_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.SAT_Riss_Solver import SAT_Riss_Solver

if __name__ == "__main__":
    ESTIMATOR_PCT = 0.5

    instances = SAT_from_index_file(
        filepath=DATA_DIR / "SAT" / "index.json",
        max_cost=100.0,
        max_time=10.0,
    )

    train_instances = instances[:15] + instances[80:95]
    test_instances = instances[15:65] + instances[95:145]
    for instance in test_instances:
        instance.max_cost = 1000.0
        instance.max_time = 100.0

    portfolio = Portfolio.from_solver_class(SAT_Riss_Solver, size=2)

    aac = AAC(
        portfolio=portfolio,
        instance_list=train_instances,
        prefix="config",
        max_iter=75,
        calculate_features=True,
        estimator=None,
    )

    last_model_iter = 0

    estimator = None
    db = DB()
    for _ in aac.configure_iter():
        if aac.iter >= 37 and aac.iter - last_model_iter >= 5:
            X, y = get_model_training_data(db)
            estimator = Estimator1(max_cost=100.0, estimator_pct=ESTIMATOR_PCT)
            estimator.fsit(X, y)
            estimator.log()
            last_model_iter = aac.iter
            aac.update(estimator=estimator)

    for i in range(10):
        portfolio.evalusate(
            test_instances,
            prefix=f"test{i}",
            calculate_features=False,
            cache=False,
        )
