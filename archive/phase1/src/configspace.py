from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    Float,
    InCondition,
    Integer,
)
from ConfigSpace.conditions import EqualsCondition, InCondition

RANDOM_STATE = 0


###################################################################################
# RIDGE
###################################################################################
RIDGE_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Constant(name="random_state", value=RANDOM_STATE),
        Float(name="alpha", bounds=(1e-6, 1e3), default=1.0, log=True),
    ],
)


###################################################################################
# POLY RIDGE
###################################################################################
POLY_RIDGE_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Float(name="alpha", bounds=(1e-6, 1e3), default=1.0, log=True),
        Constant(name="degree", value=2),
        Categorical(name="interaction_only", items=[False, True], default=False),
    ],
)

###################################################################################
# RANDOM FOREST
###################################################################################
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

###################################################################################
# XGB
###################################################################################
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

###################################################################################
# SVR
###################################################################################
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

###################################################################################
# GPR
###################################################################################
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


###################################################################################
# COX PH
###################################################################################
COX_PH_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Float(name="alpha", bounds=(1e-6, 1e3), default=1.0, log=True),
        Categorical(name="ties", items=["breslow", "efron"], default="breslow"),
    ],
)


###################################################################################
# RANDOM SURVIVAL FOREST
###################################################################################
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

###################################################################################
# GB COX
###################################################################################
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

###################################################################################
# SURVIVAL FUNCTION WRAPPER
###################################################################################
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

###################################################################################
# XGB AFT
###################################################################################
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

###################################################################################
# TOBIT NN
###################################################################################
TOBIT_NN_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Float(name="base_lr", bounds=(1e-4, 1e-2), default=1e-3, log=True),
        Integer(name="scheduler_step_size_up", bounds=(10, 1000), default=100),
        Float(name="momentum", bounds=(0.1, 0.99), default=0.99),
        Integer(name="n_epochs", bounds=(10, 500), default=250),
        Float(name="dropout", bounds=(0.0, 1.0), default=0.5),
    ],
)


###################################################################################
# SCHMEE & HAHN QUANTILE RANDOM FOREST
###################################################################################
SCHMEE_HAHN_QRF_CONFIGSPACE = ConfigurationSpace(
    seed=0,
    space=[
        Integer(name="k", bounds=(1, 10), default=5),
        Integer(name="max_depth", bounds=(2, 32), default=32),
        Integer(name="min_samples_split", bounds=(2, 32), default=2),
        Integer(name="min_samples_leaf", bounds=(1, 32), default=1),
        Float(name="max_features", bounds=(0, 1.0), default=1.0),
        Float(name="ccp_alpha", bounds=(1e-3, 10.0), default=1e-3, log=True),
        Constant(name="random_state", value=RANDOM_STATE),
        Constant(name="n_jobs", value=-1),
        Integer(name="max_depth_rf", bounds=(2, 32), default=32),
        Integer(name="min_samples_split_rf", bounds=(2, 32), default=2),
        Integer(name="min_samples_leaf_rf", bounds=(1, 32), default=1),
        Float(name="max_features_rf", bounds=(0, 1.0), default=1.0),
        Float(name="ccp_alpha_rf", bounds=(1e-3, 10.0), default=1e-3, log=True),
    ],
)
