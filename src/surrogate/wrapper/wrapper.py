import inspect
from abc import ABC, abstractmethod

import numpy as np

from src.log import logger


class BaseWrapper(ABC):
    def __init__(
        self,
        model_cls,
        **kwargs,
    ):
        valid_params = inspect.signature(model_cls.__init__).parameters.keys()
        valid_params = [param for param in valid_params if param != "self"]
        self.model_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
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


class EmptyWrapper(StandardScalerMixin, BaseWrapper):
    class EmptyModel:
        def fit(self, X, y, cut_off):
            logger.debug(f"Fitting empty model with {X.shape[0]} samples")
            pass

        def predict(self, X, cut_off):
            logger.debug(f"Predicting with empty model with {X.shape[0]} samples")
            return np.zeros(X.shape[0])

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(self.EmptyModel, **kwargs)

    def _fit(self, X, y, cut_off) -> "EmptyWrapper":
        logger.debug(f"Fitting empty model wrapper with {X.shape[0]} samples")
        self.model.fit(X, y, cut_off)
        return self

    def _predict(self, X, cut_off) -> np.ndarray:
        logger.debug(f"Predicting {X.shape[0]} samples with empty model wrapper")
        return self.model.predict(X, cut_off)
