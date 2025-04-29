import numpy as np
import pandas as pd

from src.constant import SEED
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
        return False

    def notify_iter(self, iter: int):
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

    def notify_iter(self, iter: int):
        self.log()
        solver_count = get_solvers_count(DB())
        logger.debug(f"SurrogatePolicy.notify_iter({solver_count=})")
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


class EvaluationSurrogatePolicyA(SurrogatePolicy):
    def __init__(
        self,
        estimator_wrapper: BaseWrapper,
        first_fit_solver_count: int,
        refit_solver_count: int,
        pct_chance: float,
    ):
        super().__init__(estimator_wrapper, first_fit_solver_count, refit_solver_count)
        self._rng = np.random.default_rng(SEED)
        self.pct_chance = pct_chance

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        x = self.is_fitted and self._rng.random() < self.pct_chance
        return x


class EvaluationSurrogatePolicyB(SurrogatePolicy):
    def __init__(
        self,
        estimator_wrapper: BaseWrapper,
        first_fit_solver_count: int,
        refit_solver_count: int,
        reevaluate_pct: float,
    ):
        super().__init__(estimator_wrapper, first_fit_solver_count, refit_solver_count)
        self.reevaluate_pct = reevaluate_pct
        self._costs = None
        self._records = []

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        if self._costs is None:
            if len(self._records) > 0:
                self._costs = (
                    pd.DataFrame(self._records)
                    .set_index("id")
                    .sort_values(by="cost", ascending=True)
                    .assign(i=lambda x: range(1, len(x) + 1))
                )
            else:
                return False
        n_reevaluate = int(self.reevaluate_pct * self._costs.shape[0]) + 1
        id_ = f"{solver.id()}_{instance.id()}"
        if id_ not in self._costs.index:
            return False
        i = self._costs.at[id_, "i"]
        return i <= n_reevaluate

    def notify_iter(self, iter: int):
        super().notify_iter(iter)
        self._costs = None
        self._records = []

    def digest_results(self, solver_result: "Solver.Result"):
        if solver_result.surrogate:
            self._records.append(
                {
                    "id": solver_result.evaluation_id(),
                    "cost": solver_result.cost,
                }
            )


class EvaluationSurrogatePolicyC(SurrogatePolicy):
    def __init__(
        self,
        estimator_wrapper: BaseWrapper,
        first_fit_solver_count: int,
        refit_solver_count: int,
        reevaluate_factor: float = 1.0,
    ):
        super().__init__(estimator_wrapper, first_fit_solver_count, refit_solver_count)
        self.reevaluate_factor = reevaluate_factor
        self.cut_off_time_dict = {}

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        id_ = f"{solver.id()}_{instance.id()}"
        if id_ not in self.cut_off_time_dict:
            return False
        cut_off_time = self.cut_off_time_dict[id_]
        instance.cut_off_time = cut_off_time * self.reevaluate_factor
        instance.cut_off_cost = 10 * cut_off_time * self.reevaluate_factor
        return True

    def notify_iter(self, iter: int):
        super().notify_iter(iter)
        self.cut_off_time_dict = {}

    def digest_results(self, solver_result: "Solver.Result"):
        if solver_result.surrogate:
            id_ = solver_result.evaluation_id()
            self.cut_off_time_dict[id_] = round(solver_result.cost, 2)


class IterationSurrogatePolicyA(SurrogatePolicy):
    def __init__(
        self,
        estimator_wrapper: BaseWrapper,
        first_fit_solver_count: int,
        refit_solver_count: int,
        iter_diff: int,
    ):
        super().__init__(estimator_wrapper, first_fit_solver_count, refit_solver_count)
        self.iter_diff = iter_diff
        self.iter_counter = 1

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted and self.iter_counter != self.iter_diff

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return self.iter_counter == self.iter_diff

    def notify_iter(self, iter: int):
        super().notify_iter(iter)
        if self.is_fitted:
            self.iter_counter += 1
            if self.iter_counter > self.iter_diff:
                self.iter_counter = 1


class IterationSurrogatePolicyB(SurrogatePolicy):
    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return False

    def should_reevaluate_portfolio(
        self,
        portfolio_evaluation_result: "Portfolio.Result",
        best_incumbent_cost: float,
    ):
        return self.is_fitted and best_incumbent_cost > portfolio_evaluation_result.cost
