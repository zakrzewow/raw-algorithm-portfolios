from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.constant import MAX_WORKERS, SEED
from src.log import logger


class SurrogateEstimator(BaseEstimator, RegressorMixin, ABC):
    _RNG = np.random.default_rng(SEED)

    def __init__(self, estimator_pct: float = 0.5):
        super().__init__()
        self._estimator_pct = estimator_pct

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def log(self):
        pass

    def set_estimator_pct(self, estimator_pct):
        self._estimator_pct = estimator_pct

    @classmethod
    def get_or_none(cls, estimator: "SurrogateEstimator"):
        if estimator is None:
            return None
        return estimator if cls._RNG.random() < estimator._estimator_pct else None


class Estimator1(SurrogateEstimator):
    _DEFAULT_CLASSIFIER = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=12,
        max_features=0.4,
        min_samples_leaf=0.03,
        min_samples_split=0.05,
        random_state=SEED,
        n_jobs=MAX_WORKERS,
    )

    _DEFAULT_REGRESSOR = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        max_features=0.4,
        min_samples_leaf=0.03,
        min_samples_split=0.05,
        random_state=SEED,
        n_jobs=MAX_WORKERS,
    )

    def __init__(
        self,
        max_cost: float,
        estimator_pct: float = 0.5,
        classifier: BaseEstimator = None,
        regressor: BaseEstimator = None,
    ):
        super().__init__(estimator_pct)
        self.max_cost = max_cost
        if classifier is None:
            classifier = clone(self._DEFAULT_CLASSIFIER)
        if regressor is None:
            regressor = clone(self._DEFAULT_REGRESSOR)
        self.classifier = classifier
        self.regressor = regressor
        self._is_fitted_ = False
        self._training_data_ = None
        self._mask_non_timeout = None

    def fit(self, X, y):
        y_timeout = y != self.max_cost
        self.classifier.fit(X, y_timeout)

        mask_non_timeout = y < self.max_cost
        if np.any(mask_non_timeout):
            self.regressor.fit(X[mask_non_timeout], y[mask_non_timeout])

        self._is_fitted_ = True
        self._training_data_ = X, y
        self._mask_non_timeout = mask_non_timeout
        return self

    def predict(self, X):
        check_is_fitted(self, "_is_fitted_")
        is_timeout = ~self.classifier.predict(X)
        costs_pred = np.full(X.shape[0], self.max_cost, dtype=float)

        try:
            if np.any(~is_timeout) and check_is_fitted(self.regressor):
                costs_pred[~is_timeout] = self.regressor.predict(X[~is_timeout])
        except NotFittedError:
            pass

        return costs_pred

    def log(self):
        if not self._is_fitted_:
            logger.debug("Estimator1(fitted=False)")
        else:
            X, y = self._training_data_
            score = self.score(X, y)
            training_data_shape = X.shape
            non_timeout_training_data_shape = X[self._mask_non_timeout].shape
            logger.debug(
                f"Estimator1("
                f"fitted={self._is_fitted_}, "
                f"score={score}, "
                f"estimator_pct={self._estimator_pct}, "
                f"training_data_shape={training_data_shape}, "
                f"non_timeout={non_timeout_training_data_shape}"
                f")"
            )
