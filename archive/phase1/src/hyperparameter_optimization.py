from typing import List, Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from .evaluation import evaluate_model_with_cross_validation


def optimize_hyperparameters(
    df,
    model_cls,
    wrapper_cls,
    configspace: ConfigurationSpace,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    const_cut_off: float = None,
    permuation_lognormal_mean_sigma: Tuple[float, float] = None,
    n_trials=30,
    random_state=0,
):
    def train(config: Configuration, seed) -> float:
        wrapper = wrapper_cls(model_cls=model_cls, **config)
        result = evaluate_model_with_cross_validation(
            df=df,
            wrapper=wrapper,
            splits=splits,
            const_cut_off=const_cut_off,
            permuation_lognormal_mean_sigma=permuation_lognormal_mean_sigma,
            random_state=random_state,
        )
        return result["rmse"]

    scenario = Scenario(
        configspace, deterministic=True, n_trials=n_trials, seed=random_state
    )
    smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
    incumbent = smac.optimize()
    incumbent = dict(incumbent)
    incumbent["model_cls"] = model_cls
    return incumbent
