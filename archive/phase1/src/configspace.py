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
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from xgboost import XGBRegressor

if __name__ == "__main__":
    Ridge
    RandomForestRegressor
    XGBRegressor
    SVR
    RandomSurvivalForest
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
        Constant(name="random_state", value=RANDOM_STATE),
        Constant(name="n_jobs", value=-1),
    ],
)


XGB_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Integer(name="n_estimators", bounds=(10, 1000), default=100),
        Integer(name="max_depth", bounds=(2, 10), default=6),
        Float(name="learning_rate", bounds=(0.01, 0.3), default=0.1, log=True),
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


for hp in SURVIVAL_FUNCTION_WRAPPER_CONFIGSPACE.values():
    COX_PH_CONFIGSPACE.add(hp)
    RANDOM_SURVIVAL_FOREST_CONFIGSPACE.add(hp)

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
