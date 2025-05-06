import inspect
from abc import ABC, abstractmethod

import numpy as np


class BaseWrapper(ABC):
    def __init__(
        self,
        model_cls,
        **kwargs,
    ):
        # valid_params = inspect.signature(model_cls.__init__).parameters.keys()
        # valid_params = [param for param in valid_params if param != "self"]
        # self.model_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        self.model_kwargs = kwargs
        self.model_cls = model_cls
        self.model = None

    def fit(self, X, y, cut_off) -> "BaseWrapper":
        X, y, cut_off = self._preprocess_fit(X, y, cut_off)
        self.model = self.model_cls(**self.model_kwargs)
        return self._fit(X, y, cut_off)

    def _preprocess_fit(self, X, y, cut_off):
        return X, y, cut_off

    @abstractmethod
    def _fit(self, X, y, cut_off) -> "BaseWrapper":
        pass

    def predict(self, X, cut_off) -> np.ndarray:
        X, cut_off = self._preprocess_predict(X, cut_off)
        y_pred = self._predict(X, cut_off)
        y_pred = self._postprocess_predict(y_pred)
        return y_pred

    def _preprocess_predict(self, X, cut_off):
        return X, cut_off

    @abstractmethod
    def _predict(self, X, cut_off) -> np.ndarray:
        pass

    def _postprocess_predict(self, y_pred):
        return y_pred


class StandardScalerMixin:
    def _preprocess_fit(self, X, y, cut_off):
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)
        return super()._preprocess_fit(X, y, cut_off)

    def _preprocess_predict(self, X, cut_off):
        X = self.scaler_.transform(X)
        return super()._preprocess_predict(X, cut_off)


class LogPredictMixin:
    def _preprocess_fit(self, X, y, cut_off):
        y = np.log(y + 0.01)
        cut_off = np.log(cut_off + 0.01)
        return super()._preprocess_fit(X, y, cut_off)

    def _postprocess_predict(self, y_pred):
        y_pred = np.exp(y_pred) - 0.01
        return super()._postprocess_predict(y_pred)


class SkipCutOffMixin:
    def _preprocess_fit(self, X, y, cut_off):
        idx = y < cut_off
        if idx.sum() == 0:
            raise ValueError("No data points below cut-off.")
        X = X[idx]
        y = y[idx]
        cut_off = cut_off[idx]
        return super()._preprocess_fit(X, y, cut_off)


class StandardScaledWrapper(StandardScalerMixin, BaseWrapper):
    """A wrapper that applies standard scaling to features."""

    pass


class LogTransformedWrapper(LogPredictMixin, BaseWrapper):
    """A wrapper that applies log transformation to targets."""

    pass


class StandardScaledLogTransformedWrapper(
    StandardScalerMixin,
    LogPredictMixin,
    BaseWrapper,
):
    """A wrapper that applies both standard scaling to features and log transformation to targets."""

    def _fit(self, X, y, cut_off) -> "StandardScaledLogTransformedWrapper":
        self.model.fit(X, y, cut_off)
        return self

    def _predict(self, X, cut_off) -> np.ndarray:
        return self.model.predict(X, cut_off)


class ScikitLearnWrapper(StandardScaledLogTransformedWrapper):
    """A wrapper for scikit-learn models."""

    def _fit(self, X, y, cut_off) -> "ScikitLearnWrapper":
        self.model.fit(X, y)
        return self

    def _predict(self, X, cut_off) -> np.ndarray:
        return self.model.predict(X)


class SkipCutOffScikitLearnWrapper(
    SkipCutOffMixin,
    ScikitLearnWrapper,
):
    """A wrapper for scikit-learn models that skips cutoff."""

    pass
