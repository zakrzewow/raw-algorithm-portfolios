import os

from src.constant import N_TRAIN, PARG, POLICY, SEED
from src.experiment import parhydra_bbob
from src.instance.BBOB_Instance import BBOB_test, BBOB_train, set_08_cut_off_time
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
    train_instances = BBOB_train(
        n=N_TRAIN,
        seed=SEED,
    )
    test_instances = BBOB_test()
    train_instances = set_08_cut_off_time(train_instances)

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

    portfolio = parhydra_bbob(
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
