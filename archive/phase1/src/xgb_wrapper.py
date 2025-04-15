import numpy as np
import xgboost as xgb

from .wrapper import BaseWrapper, StandardScalerMixin


class XGBwrapper(StandardScalerMixin, BaseWrapper):
    def __init__(
        self,
        model_cls,
        **kwargs,
    ):
        super().__init__(model_cls, **kwargs)

    def _fit(self, X, y, cut_off) -> "XGBwrapper":
        dtrain = xgb.DMatrix(X)
        y_lower_bound = y
        y_upper_bound = np.where(y == cut_off, np.inf, y)
        dtrain.set_float_info("label_lower_bound", y_lower_bound)
        dtrain.set_float_info("label_upper_bound", y_upper_bound)
        self.model.fit(dtrain)
        return self

    def _predict(self, X, cut_off) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
