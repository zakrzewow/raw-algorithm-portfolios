import numpy as np
import pandas as pd

from src.constant import PARG, SEED
from src.database.db import DB
from src.database.queries import get_model_training_data, get_solvers_count
from src.log import logger
from src.surrogate.model import CoxPHSurvivalAnalysis, XGBRegressorAFT
from src.surrogate.wrapper import SurvivalFunctionWrapper, XGBWrapper

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
        first_fit_solver_count: int,
        refit_solver_count: int,
    ):
        self.first_fit_solver_count = first_fit_solver_count
        self.refit_solver_count = refit_solver_count
        self.last_fit_solver_count = 0
        self.is_fitted = False
        self.estimator_wrapper = None

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
        if self.last_fit_solver_count <= 20:
            self.estimator_wrapper = SurvivalFunctionWrapper(
                model_cls=CoxPHSurvivalAnalysis,
                risk_function="polynomial",
                risk_alpha=0.55,
                ties="breslow",
                alpha=13.69,
            )
        else:
            self.estimator_wrapper = XGBWrapper(
                model_cls=XGBRegressorAFT,
                aft_loss_distribution="logistic",
                colsample_bytree=0.63,
                eval_metric="aft-nloglik",
                gamma=2.50,
                learning_rate=0.073,
                max_depth=3,
                min_child_weight=5,
                num_boost_round=627,
                objective="survival:aft",
                reg_alpha=0.006,
                reg_lambda=0.0013,
                seed=0,
                subsample=0.55,
                aft_loss_distribution_scale=0.72,
            )
        X, y, cut_off = get_model_training_data(DB())
        logger.debug(f"SurrogatePolicy.refit_estimator(X.shape={X.shape})")
        self.estimator_wrapper.fit(X, y, cut_off)


class TestSurrogatePolicy(SurrogatePolicy):
    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted and instance.tsp_generator == "cluster_netgen"


class EvaluationSurrogatePolicyA(SurrogatePolicy):
    def __init__(
        self,
        first_fit_solver_count: int,
        refit_solver_count: int,
        pct_chance: float = None,
    ):
        super().__init__(first_fit_solver_count, refit_solver_count)
        self._rng = np.random.default_rng(SEED)
        if pct_chance is None:
            pct_chance = float(PARG) / 100.0
        self.pct_chance = pct_chance

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        x = self.is_fitted and self._rng.random() < self.pct_chance
        return x


class EvaluationSurrogatePolicyB(SurrogatePolicy):
    def __init__(
        self,
        first_fit_solver_count: int,
        refit_solver_count: int,
        reevaluate_pct: float = None,
    ):
        super().__init__(first_fit_solver_count, refit_solver_count)
        if reevaluate_pct is None:
            reevaluate_pct = float(PARG) / 100.0
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
        first_fit_solver_count: int,
        refit_solver_count: int,
        reevaluate_factor: float = None,
    ):
        super().__init__(first_fit_solver_count, refit_solver_count)
        if reevaluate_factor is None:
            reevaluate_factor = float(PARG) / 100.0
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
            self.cut_off_time_dict[id_] = min(round(solver_result.cost, 2), 20.0)


class IterationSurrogatePolicyA(SurrogatePolicy):
    def __init__(
        self,
        first_fit_solver_count: int,
        refit_solver_count: int,
        surrogate_iter: int = None,
        real_iter: int = None,
    ):
        super().__init__(first_fit_solver_count, refit_solver_count)
        if surrogate_iter is None or real_iter is None:
            surrogate_iter, real_iter = PARG.split("+")
            surrogate_iter = int(surrogate_iter)
            real_iter = int(real_iter)
        self.surrogate_iter = surrogate_iter
        self.real_iter = real_iter
        self.surrogate_mode = True
        self.iter_counter = 0

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return self.is_fitted and self.surrogate_mode

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return False

    def notify_iter(self, iter: int):
        super().notify_iter(iter)
        if self.is_fitted:
            self.iter_counter += 1
            if self.surrogate_mode and self.iter_counter > self.surrogate_iter:
                self.iter_counter = 1
                self.surrogate_mode = False
            elif not self.surrogate_mode and self.iter_counter > self.real_iter:
                self.iter_counter = 1
                self.surrogate_mode = True


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
