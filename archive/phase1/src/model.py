import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
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
from tqdm.auto import tqdm
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
    def __init__(self, input_dim, init_bias_std):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.mu = nn.Linear(50, 1)
        self.log_sigma = nn.Linear(50, 1)
        self.log_sigma.bias.data.fill_(init_bias_std.log())

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
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
    def __init__(self):
        self.model = None

    def fit(self, X, y, cut_off):
        is_censored = y >= cut_off

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        is_censored = torch.tensor(is_censored, dtype=torch.float32)

        dataset = TensorDataset(X, y, is_censored)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        init_std = y.std()
        self.model = TobitNet(input_dim=X.shape[1], init_bias_std=init_std)
        optimizer = SGD(
            self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4
        )
        scheduler = CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-2,
            step_size_up=100,
            mode="triangular",
        )

        self.model.train()
        for epoch in tqdm(range(250)):
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
        y_pred, _ = self.model(X)
        y_pred = y_pred.detach().numpy()
        return y_pred
