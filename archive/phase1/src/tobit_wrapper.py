import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset


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
        for epoch in range(500):
            for X_batch, y_batch, censored_batch in loader:
                mu_pred, sigma_pred = self.model(X_batch)
                loss = tobit_loss(mu_pred, sigma_pred, y_batch, censored_batch.float())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()

        return self


# class TobitWrapper(StandardScaledLogTransformedWrapper):
#     def _fit(self, X, y, cut_off) -> "TobitWrapper":
#         self.model.fit(X, y, cut_off)
#         return self

#     def _predict(self, X, cut_off) -> np.ndarray:
#         return self.model.predict(X)
