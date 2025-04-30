import xgboost as xgb
from sksurv.linear_model import CoxPHSurvivalAnalysis

if __name__ == "__main__":
    CoxPHSurvivalAnalysis


class XGBRegressorAFT:
    def __init__(
        self,
        objective="survival:aft",
        eval_metric="aft-nloglik",
        aft_loss_distribution="normal",
        aft_loss_distribution_scale=1.0,
        num_boost_round=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1e-3,
        reg_alpha=1e-3,
        seed=0,
    ):
        self.params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "aft_loss_distribution": aft_loss_distribution,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "seed": seed,
        }
        self.num_boost_round = num_boost_round

    def fit(self, dtrain):
        self.bst = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        return self

    def predict(self, dtest):
        return self.bst.predict(dtest)
