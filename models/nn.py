from functools import partial
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
    'nn__hidden_sizes': [(32,), (64,), (64, 32), (128, 64)],
    'nn__dropout': [0.0, 0.1, 0.3],
    'nn__weight_decay': loguniform(1e-5, 1e-1),
    'nn__lr': loguniform(1e-4, 1e-2),
}

# параметры задаются строками, чтобы их можно было сохранить в json и прочитать в логе
ACTIVATIONS = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}
# без momentum SGD учится слишком медленно, чтобы сравнение с Adam было честным
OPTIMIZERS = {'adam': optim.Adam, 'adamw': optim.AdamW, 'sgd': partial(optim.SGD, momentum=0.9)}
SCHEDULERS = {'cosine': optim.lr_scheduler.CosineAnnealingLR}
LOSSES = {'mse': nn.MSELoss, 'huber': nn.HuberLoss}


def build_mlp(n_inputs, hidden_sizes, activation, batchnorm, dropout):
    '''Собирает MLP: на каждый скрытый слой Linear -> [BatchNorm] -> активация -> [Dropout], в конце один выход'''
    layers = []
    prev_size = n_inputs
    for size in hidden_sizes:
        layers.append(nn.Linear(prev_size, size))
        if batchnorm:
            layers.append(nn.BatchNorm1d(size))
        layers.append(ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_size = size
    layers.append(nn.Linear(prev_size, 1))
    return nn.Sequential(*layers)


class MLRegressor(RegressorMixin, BaseEstimator):
    """Полносвязная нейросеть для регрессии, архитектура и обучение задаются параметрами"""
    def __init__(self, hidden_sizes=(10,), activation='relu', batchnorm=False, dropout=0.0,
                 optimizer='adam', lr=0.001, weight_decay=0.0, scheduler=None, loss='mse',
                 epochs=100, batch_size=32, verbose=False):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.loss = loss
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
        # BatchNorm падает на батче из одной строки, поэтому с ним неполный последний батч отбрасывается
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True,
                            drop_last=self.batchnorm)

        self.model_ = build_mlp(X.shape[1], self.hidden_sizes, self.activation, self.batchnorm, self.dropout)
        optimizer = OPTIMIZERS[self.optimizer](self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = SCHEDULERS[self.scheduler](optimizer, T_max=self.epochs) if self.scheduler else None
        loss_fn = LOSSES[self.loss]()

        self.model_.train()
        for epoch in range(self.epochs):
            squared_error, n_seen = 0.0, 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                # для лога считаем MSE отдельно, чтобы он не зависел от выбранного loss
                squared_error += ((pred.detach() - y_batch) ** 2).sum().item()
                n_seen += len(X_batch)
            if scheduler is not None:
                scheduler.step()
            if self.verbose and epoch % 10 == 0:
                # MSE посчитан на масштабированном y, переводим в RMSE лог-цены
                rmse = np.sqrt(squared_error / n_seen) * self.y_std_
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
