from typing import Literal

import numpy as np

from .wrapper import BaseWrapper, StandardScalerMixin


class SurvivalFunctionWrapper(StandardScalerMixin, BaseWrapper):
    def __init__(
        self,
        model_cls,
        risk_function: Literal[
            "linear",
            "polynomial",
            "exponential",
            "par10",
        ] = "linear",
        risk_alpha: float = None,
        risk_beta: float = None,
        **kwargs,
    ):
        super().__init__(model_cls, **kwargs)

        self.risk_function = risk_function
        self.risk_alpha = risk_alpha
        self.risk_beta = risk_beta
        if risk_function not in ["linear", "polynomial", "exponential", "par10"]:
            raise ValueError(
                "Invalid risk function. Choose from 'linear', 'polynomial', 'exponential', 'par10'."
            )
        if risk_function == "polynomial" and risk_alpha is None:
            raise ValueError("Alpha must be provided for polynomial risk function.")
        if risk_function == "exponential" and (risk_alpha is None or risk_beta is None):
            raise ValueError(
                "Alpha and beta must be provided for exponential risk function."
            )

    def _fit(self, X, y, cut_off) -> "SurvivalFunctionWrapper":
        y_ndarr = np.zeros(X.shape[0], dtype=[("not_censored", bool), ("cost", float)])
        y_ndarr["not_censored"] = y < cut_off
        y_ndarr["cost"] = y
        self.model.fit(X, y_ndarr)
        return self

    def _predict(self, X, cut_off) -> np.ndarray:
        sf_arr = self.model.predict_survival_function(X, return_array=True)
        sf_avg = 0.5 * (sf_arr[:, :-1] + sf_arr[:, 1:])
        t = self.model.unique_times_.copy()
        if self.risk_function == "linear":
            pass
        elif self.risk_function == "polynomial":
            t = t**self.risk_alpha
        elif self.risk_function == "exponential":
            t = np.clip(t / t.max(), 0, 1 - 1e-3)
            t = np.minimum((-1) * self.risk_alpha * np.log(1.0 - t), self.risk_beta)
        elif self.risk_function == "par10":
            t = np.tile(t, (sf_avg.shape[0], 1))
            cut_off_2d = np.tile(cut_off[:, np.newaxis], (1, t.shape[1]))
            t[t >= cut_off_2d] *= 10
        dt = np.diff(t)
        return np.sum(sf_avg * dt, axis=1)
