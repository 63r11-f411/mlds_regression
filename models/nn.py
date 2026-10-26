from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from config import RANDOM_STATE
from sklearn.pipeline import Pipeline
from preprocessing import build_scaled_preprocessor
from scipy.stats import loguniform

PARAM_GRID = {
    'nn__hidden_size': [8, 16, 32, 64, 128],
    'nn__lr': loguniform(1e-4, 1e-2),
    'nn__epochs': [50, 100, 200],
    'nn__batch_size': [16, 32, 64, 128]
}

class MLRegressor(RegressorMixin, BaseEstimator):
    """Простая нейронная сеть для регрессии с одним скрытым слоем"""
    def __init__(self, hidden_size=10, lr=0.001, epochs=100, batch_size=32, verbose=False):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose


    def fit(self, X, y):
        torch.manual_seed(RANDOM_STATE)

        y = np.asarray(y)
        self.y_mean_ = y.mean()
        self.y_std_ = y.std()
        y_scaled = (y - self.y_mean_) / self.y_std_

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        self.model_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            if self.verbose and epoch % 10 == 0:
                # loss посчитан на масштабированном y, переводим в RMSE лог-цены
                rmse = np.sqrt(epoch_loss / len(X_tensor)) * self.y_std_
                print(f'epoch {epoch}: train RMSE {rmse:.4f}')

        return self

    def predict(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        return predictions.numpy().reshape(-1) * self.y_std_ + self.y_mean_


def get_nn_pipeline(preprocessor=None, **nn_params):
    if preprocessor is None:
        preprocessor = build_scaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('nn', MLRegressor(**nn_params))
    ])
