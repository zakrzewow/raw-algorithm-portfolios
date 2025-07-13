import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from .evaluation import evaluate_model_with_cross_validation


def read_pickle(filepath: Path):
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)


def to_pickle(filepath: Path, obj):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def optimize_hyperparameters(
    df,
    model_cls,
    wrapper_cls,
    configspace: ConfigurationSpace,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    const_cut_off: float = None,
    permuation_lognormal_mean_sigma: Tuple[float, float] = None,
    instance_to_cut_off: dict = None,
    n_trials=30,
    random_state=0,
    filepath: Path = None,
):
    if filepath is not None:
        if (incumbent := read_pickle(filepath)) is not None:
            return incumbent

    def train(config: Configuration, seed) -> float:
        wrapper = wrapper_cls(model_cls=model_cls, **config)
        result = evaluate_model_with_cross_validation(
            df=df,
            wrapper=wrapper,
            splits=splits,
            const_cut_off=const_cut_off,
            permuation_lognormal_mean_sigma=permuation_lognormal_mean_sigma,
            instance_to_cut_off=instance_to_cut_off,
            random_state=random_state,
        )
        return result["rmse"]

    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=n_trials,
        seed=random_state,
        use_default_config=True,
    )
    smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
    incumbent = smac.optimize()
    incumbent = dict(incumbent)
    incumbent["model_cls"] = model_cls

    if filepath is not None:
        to_pickle(filepath, incumbent)

    return incumbent
