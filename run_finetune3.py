from src.constant import DATA_DIR
from src.experiment.FineTuneExperiment import FineTuneExperiment
from src.instance.TSP_Instance import TSP_Instance, TSP_InstanceSet
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=30,
        seed=0,
    )
    solver_class = TSP_LKH_Solver
    instance_class = TSP_Instance
    experiment = FineTuneExperiment(
        db_name="MY_CEPS1.db",
        solver_class=solver_class,
        instance_class=instance_class,
    )
    best_portfolio = experiment.construct_portfolio(train_instances)
    experiment.fine_tune_all(best_portfolio, test_instances, comment="finetune1")
