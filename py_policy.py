import os

from src.constant import DATA_DIR, N_TRAIN, PARG, POLICY, SEED
from src.experiment import parhydra
from src.instance.TSP_Instance import TSP_from_index_file, set_n22_cut_off_time
from src.log import logger
from src.surrogate.SurrogatePolicy import (
    EmptySurrogatePolicy,
    EvaluationSurrogatePolicyA,
    EvaluationSurrogatePolicyB,
    EvaluationSurrogatePolicyC,
    IterationSurrogatePolicyA,
    IterationSurrogatePolicyB,
)

if __name__ == "__main__":
    logger.info(f"{POLICY=}, {PARG=}, {N_TRAIN=}, {SEED=}")
    train_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN" / "index.json",
        cut_off_cost=100,
        cut_off_time=10,
        n=N_TRAIN,
        seed=SEED,
    )
    test_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TEST" / "index.json",
        cut_off_cost=1000,
        cut_off_time=100,
        n=250,
        seed=0,
    )
    train_instances = set_n22_cut_off_time(train_instances, reference_cut_off_time=10.0)

    POLICY_KWARGS = {
        "first_fit_solver_count": 5,
        "refit_solver_count": 5,
    }

    POLICY = os.environ.get("POLICY", "").strip()
    if POLICY == "ea":
        surrogate_policy = EvaluationSurrogatePolicyA(**POLICY_KWARGS)
    elif POLICY == "eb":
        surrogate_policy = EvaluationSurrogatePolicyB(**POLICY_KWARGS)
    elif POLICY == "ec":
        surrogate_policy = EvaluationSurrogatePolicyC(**POLICY_KWARGS)
    elif POLICY == "ia":
        surrogate_policy = IterationSurrogatePolicyA(**POLICY_KWARGS)
    elif POLICY == "ib":
        surrogate_policy = IterationSurrogatePolicyB(**POLICY_KWARGS)
    else:
        surrogate_policy = EmptySurrogatePolicy()

    SOLVERS_N = 2
    ATTEMPTS_N = 4
    MAX_ITER = 25

    portfolio = parhydra(
        train_instances=train_instances,
        surrogate_policy=surrogate_policy,
        SOLVERS_N=SOLVERS_N,
        ATTEMPTS_N=ATTEMPTS_N,
        MAX_ITER=MAX_ITER,
    )
    for i in range(5):
        portfolio.evaluate(
            test_instances,
            prefix=f"test{i}",
            calculate_features=False,
            cache=False,
        )
