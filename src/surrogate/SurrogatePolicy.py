from src.database.db import DB
from src.database.queries import get_model_training_data, get_solvers_count
from src.log import logger
from src.surrogate.wrapper import BaseWrapper

if __name__ == "__main__":
    from src.instance.Instance import Instance
    from src.solver.Portfolio import Portfolio
    from src.solver.Solver import Solver


class EmptySurrogatePolicy:
    def __repr__(self):
        return f"EmptySurrogatePolicy()"

    def log(self):
        logger.debug(self.__repr__())

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return False

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return True

    def notify_iter(self):
        pass

    def refit_estimator(self):
        pass

    def should_reevaluate_portfolio(
        self,
        portfolio_evaluation_result: "Portfolio.Result",
        best_incumbent_cost: float,
    ):
        return False

    def digest_results(self, solver_result: "Solver.Result"):
        pass


class SurrogatePolicy(EmptySurrogatePolicy):
    def __init__(
        self,
        estimator_wrapper: BaseWrapper,
        first_fit_solver_count: int,
        refit_solver_count: int,
    ):
        self.estimator_wrapper = estimator_wrapper
        self.first_fit_solver_count = first_fit_solver_count
        self.refit_solver_count = refit_solver_count
        self.last_fit_solver_count = 0
        self.is_fitted = False
        self._iter = 0

    def __repr__(self):
        str_ = (
            f"SurrogatePolicy("
            f"estimator_wrapper={self.estimator_wrapper}, "
            f"first_fit_solver_count={self.first_fit_solver_count}, "
            f"refit_solver_count={self.refit_solver_count}, "
            f"last_fit_solver_count={self.last_fit_solver_count}, "
            f"is_fitted={self.is_fitted})"
        )
        return str_

    def notify_iter(self):
        self._iter += 1
        self.log()
        solver_count = get_solvers_count(DB())
        logger.debug(f"SurrogatePolicy.notify_iter(iter={self._iter}, {solver_count=})")
        if (
            self.last_fit_solver_count == 0
            and solver_count >= self.first_fit_solver_count
        ) or (
            self.last_fit_solver_count > 0
            and solver_count - self.last_fit_solver_count >= self.refit_solver_count
        ):
            self.last_fit_solver_count = solver_count
            self.refit_estimator()

    def refit_estimator(self):
        self.is_fitted = True
        X, y, cut_off = get_model_training_data(DB())
        logger.debug(f"SurrogatePolicy.refit_estimator(X.shape={X.shape})")
        self.estimator_wrapper.fit(X, y, cut_off)


class TestSurrogatePolicy(SurrogatePolicy):
    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted and instance.tsp_generator == "cluster_netgen"

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return False


class IterationSurrogatePolicyB(SurrogatePolicy):
    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return True

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return False

    def should_reevaluate_portfolio(
        self,
        portfolio_evaluation_result: "Portfolio.Result",
        best_incumbent_cost: float,
    ):
        return best_incumbent_cost > portfolio_evaluation_result.cost
