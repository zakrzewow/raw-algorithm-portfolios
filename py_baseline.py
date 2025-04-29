import numpy as np

from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_from_index_file
from src.log import logger
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN" / "index.json",
        cut_off_cost=100,
        cut_off_time=10,
        n=25,
    )
    test_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TEST" / "index.json",
        cut_off_cost=1000,
        cut_off_time=100,
        n=250,
    )
    for instance in train_instances:
        instance.cut_off_time = round(10 * ((instance.n_cities / 600) ** 2.2), 2)
        instance.cut_off_cost = 10 * instance.cut_off_time

    SOLVERS_N = 2
    ATTEMPTS_N = 3
    MAX_ITER = 25

    solvers = []
    largest_marginal_contribution_solver = None

    for solver_i in range(SOLVERS_N):
        logger.info(f"Solver {solver_i + 1}/{SOLVERS_N}")

        best_cost = np.inf
        best_solver = None
        attempt_solvers = []

        for attempt_i in range(ATTEMPTS_N):
            logger.info(f"Attempt {attempt_i + 1}/{ATTEMPTS_N}")

            if largest_marginal_contribution_solver is not None:
                new_solver = largest_marginal_contribution_solver.copy()
            else:
                new_solver = TSP_LKH_Solver()

            iteration_solvers = solvers + [new_solver]

            portfolio = Portfolio.from_iterable(iteration_solvers)
            aac = AAC(
                portfolio=portfolio,
                instance_list=train_instances,
                prefix=f"config;solver={solver_i+1};attempt={attempt_i+1}",
                max_iter=MAX_ITER,
                i=solver_i,
                calculate_features=False,
            )
            portfolio = aac.configure()
            result = portfolio.evaluate(  # fix cut-off times before validation
                instance_list=train_instances,
                prefix=f"validate;solver={solver_i+1};attempt={attempt_i+1}",
                cache=True,
            )
            attempt_solvers.append(portfolio[solver_i])
            logger.info(
                f"Attempt {attempt_i + 1}/{ATTEMPTS_N}: cost = {result.cost:.2f}"
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_solver = portfolio[solver_i]

        solvers.append(best_solver)
        logger.info(f"Solver {solver_i + 1}/{SOLVERS_N}: best cost = {best_cost:.2f}")

        if solver_i < SOLVERS_N - 1:
            largest_marginal_contribution_solver = None
            best_cost = np.inf
            for attempt_i, solver in enumerate(attempt_solvers):
                if solver != best_solver:
                    portfolio = Portfolio.from_iterable(solvers + [solver])
                    result = portfolio.evaluate(
                        instance_list=train_instances,
                        prefix=f"largest_marginal_contribution;solver={solver_i+1};attempt={attempt_i+1}",
                        cache=True,
                    )
                    if result.cost < best_cost:
                        best_cost = result.cost
                        largest_marginal_contribution_solver = solver

    portfolio = Portfolio.from_iterable(solvers)
    for i in range(5):
        portfolio.evaluate(
            test_instances,
            prefix=f"test{i}",
            calculate_features=False,
            cache=False,
        )
