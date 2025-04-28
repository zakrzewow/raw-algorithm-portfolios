if __name__ == "__main__":
    from src.instance.Instance import Instance
    from src.solver.Portfolio import Portfolio
    from src.solver.Solver import Solver


class EmptySurrogatePolicy:
    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return False

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return True

    def notify_iter(self, iter: int, max_iter: int):
        pass

    def should_reevaluate_portfolio(
        self,
        portfolio_evaluation_result: "Portfolio.Result",
        best_incumbent_cost: float,
    ):
        return False

    def get_estimator(self):
        return None

    def digest_results(self, solver_result: "Solver.Result"):
        pass


class SurrogatePolicy(EmptySurrogatePolicy):
    def __init__(
        self,
        surrogate_estimator_class: type,
    ):
        self.surrogate_estimator_class = surrogate_estimator_class


class IterationSurrogatePolicyA(SurrogatePolicy):
    def __init__(
        self,
        surrogate_estimator_class: type,
        iter_lag: int = 1,
    ):
        super().__init__(surrogate_estimator_class)
        self.iter_lag = iter_lag

    def should_estimate(self, solver: "Solver", instance: "Instance"):
        return False

    def should_reevaluate(self, solver: "Solver", instance: "Instance"):
        return True

    def notify_iter(self, iter: int, max_iter: int):
        pass

    def should_reevaluate_portfolio(
        self,
        portfolio_evaluation_result: "Portfolio.Result",
        best_incumbent_cost: float,
    ):
        return False

    def get_estimator(self):
        return None

    def digest_results(self, solver_result: "Solver.Result"):
        pass


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
