import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*algorithm did not converge.*")

if __name__ == "__main__":
    Ridge
    RandomForestRegressor
    XGBRegressor
    SVR
    RandomSurvivalForest
    GradientBoostingSurvivalAnalysis
    CoxPHSurvivalAnalysis


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
                ("ridge", Ridge(alpha=self.alpha, random_state=0)),
            ]
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class GPRWithRBF(GaussianProcessRegressor):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), alpha=1e-10):
        kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        super().__init__(kernel=kernel, alpha=alpha, random_state=0)

    @property
    def length_scale(self):
        return self.kernel.length_scale

    @property
    def length_scale_bounds(self):
        return self.kernel.length_scale_bounds


class XGBRegressorAFT:
    def __init__(
        self,
        objective="survival:aft",
        eval_metric="aft-nloglik",
        aft_loss_distribution="normal",
        aft_loss_distribution_scale=1.0,
        num_boost_round=100,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_lambda=1e-3,
        reg_alpha=1e-3,
        seed=0,
    ):
        self.params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "aft_loss_distribution": aft_loss_distribution,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "seed": seed,
        }
        self.num_boost_round = num_boost_round

    def fit(self, dtrain):
        self.bst = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        return self

    def predict(self, dtest):
        return self.bst.predict(dtest)


class TobitNet(nn.Module):
    def __init__(self, input_dim, init_bias_std, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.mu = nn.Linear(50, 1)
        self.log_sigma = nn.Linear(50, 1)
        self.log_sigma.bias.data.fill_(init_bias_std.log())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        x = self.dropout(x)
        mu = self.mu(x).squeeze(-1)
        sigma = F.softplus(self.log_sigma(x)).squeeze(-1)
        return mu, sigma


def tobit_loss(mu, sigma, y, is_censored):
    eps = 1e-6
    z = (y - mu) / (sigma + eps)
    log_pdf = -0.5 * (z**2 + torch.log(2 * torch.pi * (sigma**2 + eps)))
    log_sf = torch.log(1 - 0.5 * (1 + torch.erf(z / 2**0.5)) + eps)
    loss = -((1 - is_censored) * log_pdf + is_censored * log_sf)
    return loss.mean()


class TobitModel:
    def __init__(
            self, 
            base_lr: float = 1e-3,
            scheduler_step_size_up: int = 100,
            momentum: float = 0.99,
            n_epochs: int = 250,
            dropout: float = 0.5,
        ):
        self.base_lr = base_lr
        self.scheduler_step_size_up = scheduler_step_size_up
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.model = None

    def fit(self, X, y, cut_off):
        is_censored = y >= cut_off

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        is_censored = torch.tensor(is_censored, dtype=torch.float32)

        dataset = TensorDataset(X, y, is_censored)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        init_std = y.std()
        self.model = TobitNet(input_dim=X.shape[1], init_bias_std=init_std, dropout=self.dropout)
        optimizer = SGD(
            self.model.parameters(), lr=self.base_lr, momentum=self.momentum, weight_decay=1e-4
        )
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.base_lr,
            max_lr=1e-2,
            step_size_up=self.scheduler_step_size_up,
            mode="triangular",
        )

        self.model.train()
        for epoch in range(self.n_epochs):
            for X_batch, y_batch, censored_batch in loader:
                mu_pred, sigma_pred = self.model(X_batch)
                loss = tobit_loss(mu_pred, sigma_pred, y_batch, censored_batch.float())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()

        return self

    def predict(self, X, cut_off):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        y_pred, _ = self.model(X)
        y_pred = y_pred.detach().numpy()
        return y_pred


def truncated_normal_mean(mu, sigma, C):
    sigma = np.where(sigma > 1e-5, sigma, 1e-5)
    alpha = (C - mu) / sigma
    denominator = 1 - norm.cdf(alpha)
    denominator = np.where(denominator > 1e-5, denominator, 1e-5)
    trunc_mean = mu + sigma * norm.pdf(alpha) / denominator
    return trunc_mean


class SchmeeHahnQRF:
    def __init__(
        self,
        k=10,
        max_depth=32,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        ccp_alpha=0.001,
        random_state=0,
        n_jobs=-1,
        max_depth_rf=32,
        min_samples_split_rf=2,
        min_samples_leaf_rf=1,
        max_features_rf=1.0,
        ccp_alpha_rf=0.001,
    ):
        self.k = k
        self.params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.params_rf = {
            "max_depth": max_depth_rf,
            "min_samples_split": min_samples_split_rf,
            "min_samples_leaf": min_samples_leaf_rf,
            "max_features": max_features_rf,
            "ccp_alpha": ccp_alpha_rf,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    def fit(self, X, y, cut_off):
        quantiles = np.linspace(0.01, 0.99, 99).tolist()
        not_censored = y < cut_off

        self.qrf = RandomForestQuantileRegressor(**self.params)
        self.qrf.fit(X[not_censored], y[not_censored])

        y_imputed = y.copy()

        for i in range(self.k):
            Y_pred = self.qrf.predict(X[~not_censored], quantiles=quantiles)
            y_pred_mean = Y_pred.mean(axis=1)
            y_pred_std = (Y_pred[:, 83] - Y_pred[:, 15]) / 2
            y_imputed[~not_censored] = truncated_normal_mean(
                y_pred_mean, y_pred_std, cut_off[~not_censored]
            )

            self.qrf = RandomForestQuantileRegressor(**self.params)
            self.qrf.fit(X, y_imputed)

        self.rf = RandomForestRegressor(**self.params_rf)
        self.rf.fit(X, y_imputed)
        return self

    def predict(self, X, cut_off):
        # return self.qrf.predict(X, quantiles=0.5)
        return self.rf.predict(X)
