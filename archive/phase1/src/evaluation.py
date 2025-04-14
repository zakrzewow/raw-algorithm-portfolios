from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error

from .split import permutate_df_by_cost_decreasing
from .wrapper import BaseWrapper


def evaluate_model_with_cross_validation(
    df,
    wrapper: BaseWrapper,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    const_cut_off: float = None,
    permuation_lognormal_mean_sigma: Tuple[float, float] = None,
    random_state: int = 0,
):
    result = {"wrapper": wrapper, "rmse_values": []}
    all_y_test = []
    all_y_test_not_censored = []
    all_y_pred = []

    for train_idx, test_idx in splits:
        not_train_cols = ["solver_id", "instance_id", "generator", "cost"]

        df_train = df.loc[train_idx]
        df_test = df.loc[test_idx]

        if permuation_lognormal_mean_sigma is not None:
            mean, sigma = permuation_lognormal_mean_sigma
            df_train, cut_off_train = permutate_df_by_cost_decreasing(
                df_train,
                lognormal_mean=mean,
                lognormal_sigma=sigma,
                random_state=random_state,
            )
            df_test, cut_off_test = permutate_df_by_cost_decreasing(
                df_test,
                lognormal_mean=mean,
                lognormal_sigma=sigma,
                random_state=random_state,
            )

        X_train = df_train.drop(columns=not_train_cols)
        y_train = df_train["cost"].to_numpy()

        X_test = df_test.drop(columns=not_train_cols)
        y_test = df_test["cost"].to_numpy()
        y_test_not_censored = y_test.copy()

        if permuation_lognormal_mean_sigma is not None:
            pass
        elif const_cut_off is not None:
            cut_off_train = np.full(X_train.shape[0], const_cut_off)
            cut_off_test = np.full(X_test.shape[0], const_cut_off)
        else:
            cut_off_train = np.full(X_train.shape[0], np.inf)
            cut_off_test = np.full(X_test.shape[0], np.inf)

        y_train = np.clip(y_train, 0, cut_off_train)
        y_test = np.clip(y_test, 0, cut_off_test)

        wrapper.fit(X_train, y_train, cut_off_train)
        y_pred = wrapper.predict(X_test, cut_off_test)
        y_pred = np.clip(y_pred, 0, 300.0)

        all_y_test.append(y_test)
        all_y_test_not_censored.append(y_test_not_censored)
        all_y_pred.append(y_pred)

        rmse = np.sqrt(mean_squared_error(np.log(y_test + 0.01), np.log(y_pred + 0.01)))
        result["rmse_values"].append(rmse)

    result["rmse"] = np.mean(result["rmse_values"])
    result["y_test"] = np.concatenate(all_y_test)
    result["y_test_not_censored"] = np.concatenate(all_y_test_not_censored)
    result["y_pred"] = np.concatenate(all_y_pred)

    return result
