from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from config import RANDOM_STATE
from sklearn.pipeline import Pipeline
from preprocessing import build_fe_scaled_preprocessor
from scipy.stats import loguniform

PARAM_GRID = {
    'nn__hidden_size': [8, 16, 32, 64, 128],
    'nn__lr': loguniform(1e-4, 1e-2),
    'nn__epochs': [50, 100, 200],
    'nn__batch_size': [16, 32, 64, 128]
}

class TitanicNN(ClassifierMixin, BaseEstimator):
    """Простая нейронная сеть для классификации с одним скрытым слоем"""
    def __init__(self, hidden_size=10, lr=0.001, epochs=100, batch_size=32, restart_period=5):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.restart_period = restart_period


    def fit(self, X, y):
        torch.manual_seed(RANDOM_STATE)
        self.classes_ = np.unique(y)

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        self.model_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.restart_period)

        self.model_.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            predictions = torch.sigmoid(self.model_(X_tensor))
        predictions = predictions.numpy()
        probabilities_class_1 = predictions.reshape(-1)
        probabilities_class_0 = 1.0 - probabilities_class_1
        return np.column_stack((probabilities_class_0, probabilities_class_1))


    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

def get_nn_pipeline(preprocessor=None, **nn_params):
    if preprocessor is None:
        preprocessor = build_fe_scaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('nn', TitanicNN(**nn_params))
    ])
