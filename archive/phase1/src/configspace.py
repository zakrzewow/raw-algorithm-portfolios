import xgboost as xgb
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    Float,
    InCondition,
    Integer,
)
from ConfigSpace.conditions import EqualsCondition, InCondition
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from xgboost import XGBRegressor

if __name__ == "__main__":
    Ridge
    RandomForestRegressor
    XGBRegressor
    SVR
    RandomSurvivalForest
    GradientBoostingSurvivalAnalysis
    CoxPHSurvivalAnalysis

RANDOM_STATE = 0

RIDGE_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Constant(name="random_state", value=RANDOM_STATE),
        Float(name="alpha", bounds=(1e-6, 1e3), default=1.0, log=True),
    ],
)


class PolynomialRidge:
    def __init__(self, alpha=1.0, degree=2, interaction_only=False):
        self.alpha = alpha
        self.degree = degree
        self.interaction_only = interaction_only

        self.pipeline = Pipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(
                        degree=self.degree, interaction_only=self.interaction_only
                    ),
                ),
                ("ridge", Ridge(alpha=self.alpha, random_state=RANDOM_STATE)),
            ]
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


POLY_RIDGE_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Float(name="alpha", bounds=(1e-6, 1e3), default=1.0, log=True),
        Constant(name="degree", value=2),
        Categorical(name="interaction_only", items=[False, True], default=False),
    ],
)


RANDOM_FOREST_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Integer(name="max_depth", bounds=(2, 32), default=32),
        Integer(name="min_samples_split", bounds=(2, 32), default=2),
        Integer(name="min_samples_leaf", bounds=(1, 32), default=1),
        Float(name="max_features", bounds=(0, 1.0), default=1.0),
        Float(name="ccp_alpha", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Constant(name="random_state", value=RANDOM_STATE),
        Constant(name="n_jobs", value=-1),
    ],
)


XGB_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Integer(name="n_estimators", bounds=(10, 1000), default=100),
        Integer(name="max_depth", bounds=(2, 15), default=6),
        Float(name="learning_rate", bounds=(0.001, 0.3), default=0.1, log=True),
        Float(name="subsample", bounds=(0.5, 1.0), default=1.0),
        Float(name="colsample_bytree", bounds=(0.5, 1.0), default=1.0),
        Integer(name="min_child_weight", bounds=(1, 10), default=1),
        Float(name="gamma", bounds=(0, 5), default=0),
        Float(name="reg_lambda", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Float(name="reg_alpha", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Constant(name="seed", value=RANDOM_STATE),
    ],
)


SVR_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Categorical(name="kernel", items=["poly", "rbf", "sigmoid"], default="rbf"),
        Integer(name="degree", bounds=(1, 5), default=3),
        Categorical(name="gamma", items=["scale", "auto"], default="scale"),
        Float(name="tol", bounds=(1e-3, 1e-2), log=True, default=1e-3),
        Float(name="C", bounds=(0.1, 100.0), log=True, default=1.0),
        Constant(name="max_iter", value=100000),
    ],
)
SVR_CONFIGSPACE.add(
    EqualsCondition(SVR_CONFIGSPACE["degree"], SVR_CONFIGSPACE["kernel"], "poly")
)
SVR_CONFIGSPACE.add(
    InCondition(
        SVR_CONFIGSPACE["gamma"], SVR_CONFIGSPACE["kernel"], ["rbf", "poly", "sigmoid"]
    )
)


class GPRWithRBF(GaussianProcessRegressor):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), alpha=1e-10):
        kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        super().__init__(kernel=kernel, alpha=alpha, random_state=RANDOM_STATE)

    @property
    def length_scale(self):
        return self.kernel.length_scale

    @property
    def length_scale_bounds(self):
        return self.kernel.length_scale_bounds


GPR_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Float(name="length_scale", bounds=(1e-2, 10.0), log=True, default=1.0),
        Categorical(
            name="length_scale_bounds",
            items=["fixed", (1e-5, 1e5)],
            default=(1e-5, 1e5),
        ),
        Float(name="alpha", bounds=(1e-10, 1e-1), log=True, default=1e-10),
    ],
)


SURVIVAL_FUNCTION_WRAPPER_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Categorical(
            name="risk_function",
            items=["linear", "polynomial", "exponential", "par10"],
            default="linear",
        ),
        Float(name="risk_alpha", bounds=(0.1, 10.0), default=1.0, log=False),
        Float(name="risk_beta", bounds=(0.01, 300.0), default=1.0, log=True),
    ],
)


COX_PH_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Float(name="alpha", bounds=(1e-6, 1e3), default=1.0, log=True),
        Categorical(name="ties", items=["breslow", "efron"], default="breslow"),
    ],
)


RANDOM_SURVIVAL_FOREST_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Integer(name="max_depth", bounds=(2, 32), default=32),
        Integer(name="min_samples_split", bounds=(2, 32), default=2),
        Integer(name="min_samples_leaf", bounds=(1, 32), default=1),
        Float(name="max_features", bounds=(0, 1.0), default=1.0),
        Constant(name="random_state", value=RANDOM_STATE),
        Constant(name="n_jobs", value=-1),
    ],
)


GB_COX_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Constant(name="loss", value="coxph"),
        Float(name="learning_rate", bounds=(0.001, 0.3), default=0.1, log=True),
        Integer(name="n_estimators", bounds=(10, 1000), default=100),
        Float(name="subsample", bounds=(0.5, 1.0), default=1.0),
        Integer(name="min_samples_split", bounds=(2, 32), default=2),
        Integer(name="min_samples_leaf", bounds=(1, 32), default=1),
        Integer(name="max_depth", bounds=(2, 32), default=32),
        Float(name="max_features", bounds=(0, 1.0), default=1.0),
        Float(name="ccp_alpha", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Constant(name="random_state", value=RANDOM_STATE),
    ],
)

for hp in SURVIVAL_FUNCTION_WRAPPER_CONFIGSPACE.values():
    COX_PH_CONFIGSPACE.add(hp)
    RANDOM_SURVIVAL_FOREST_CONFIGSPACE.add(hp)
    GB_COX_CONFIGSPACE.add(hp)

COX_PH_CONFIGSPACE.add(
    InCondition(
        COX_PH_CONFIGSPACE["risk_alpha"],
        COX_PH_CONFIGSPACE["risk_function"],
        ["polynomial", "exponential"],
    )
)
COX_PH_CONFIGSPACE.add(
    EqualsCondition(
        COX_PH_CONFIGSPACE["risk_beta"],
        COX_PH_CONFIGSPACE["risk_function"],
        "exponential",
    )
)
RANDOM_SURVIVAL_FOREST_CONFIGSPACE.add(
    InCondition(
        RANDOM_SURVIVAL_FOREST_CONFIGSPACE["risk_alpha"],
        RANDOM_SURVIVAL_FOREST_CONFIGSPACE["risk_function"],
        ["polynomial", "exponential"],
    )
)
RANDOM_SURVIVAL_FOREST_CONFIGSPACE.add(
    EqualsCondition(
        RANDOM_SURVIVAL_FOREST_CONFIGSPACE["risk_beta"],
        RANDOM_SURVIVAL_FOREST_CONFIGSPACE["risk_function"],
        "exponential",
    )
)
GB_COX_CONFIGSPACE.add(
    InCondition(
        GB_COX_CONFIGSPACE["risk_alpha"],
        GB_COX_CONFIGSPACE["risk_function"],
        ["polynomial", "exponential"],
    )
)
GB_COX_CONFIGSPACE.add(
    EqualsCondition(
        GB_COX_CONFIGSPACE["risk_beta"],
        GB_COX_CONFIGSPACE["risk_function"],
        "exponential",
    )
)


XGB_AFT_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Constant(name="objective", value="survival:aft"),
        Constant(name="eval_metric", value="aft-nloglik"),
        Categorical(
            name="aft_loss_distribution",
            items=["normal", "logistic", "extreme"],
            default="normal",
        ),
        Float(name="aft_loss_distribution_scale", bounds=(0.1, 2.0), default=1.0),
        Integer(name="num_boost_round", bounds=(10, 1000), default=100),
        Integer(name="max_depth", bounds=(2, 15), default=6),
        Float(name="learning_rate", bounds=(0.001, 0.3), default=0.1, log=True),
        Float(name="subsample", bounds=(0.5, 1.0), default=1.0),
        Float(name="colsample_bytree", bounds=(0.5, 1.0), default=1.0),
        Integer(name="min_child_weight", bounds=(1, 10), default=1),
        Float(name="gamma", bounds=(0, 5), default=0),
        Float(name="reg_lambda", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Float(name="reg_alpha", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Constant(name="seed", value=0),
    ],
)
XGB_AFT_CONFIGSPACE.add(
    InCondition(
        XGB_AFT_CONFIGSPACE["aft_loss_distribution_scale"],
        XGB_AFT_CONFIGSPACE["aft_loss_distribution"],
        ["normal", "logistic"],
    )
)


class XGBRegressorAFT:
    def __init__(
        self,
        objective="survival:aft",
        eval_metric="aft-nloglik",
        aft_loss_distribution="normal",
        aft_loss_distribution_scale=1.0,
        num_boost_round=100,
        n_estimators=100,
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
            "n_estimators": n_estimators,
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
