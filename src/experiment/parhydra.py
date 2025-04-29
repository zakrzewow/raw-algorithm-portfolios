import numpy as np

from src.aac.AAC import AAC
from src.instance.TSP_Instance import set_n22_cut_off_time
from src.log import logger
from src.solver.Portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver


def parhydra(
    train_instances,
    surrogate_policy,
    SOLVERS_N,
    ATTEMPTS_N,
    MAX_ITER,
):
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
                calculate_features=True,
                surrogate_policy=surrogate_policy,
            )
            portfolio = aac.configure()
            set_n22_cut_off_time(train_instances, reference_cut_off_time=10.0)
            result = portfolio.evaluate(
                instance_list=train_instances,
                prefix=f"validate;solver={solver_i+1};attempt={attempt_i+1}",
                calculate_features=True,
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
                        prefix=f"largest_marginal_contribution;attempt={attempt_i+1}",
                        calculate_features=True,
                        cache=True,
                    )
                    if result.cost < best_cost:
                        best_cost = result.cost
                        largest_marginal_contribution_solver = solver

    portfolio = Portfolio.from_iterable(solvers)
    return portfolio
