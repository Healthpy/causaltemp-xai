"""
LSTM classifier — bidirectional LSTM with attention pooling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import TSClassifier


class _LSTMModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, T) -> transpose to (N, T, M)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)          # (N, T, 2*hidden)
        scores = torch.softmax(self.attn(out), dim=1)   # (N, T, 1)
        pooled = (scores * out).sum(dim=1)              # (N, 2*hidden)
        return self.head(pooled)


class LSTMClassifier(TSClassifier):
    def __init__(self, hidden: int = 64, n_layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, epochs=50, lr=1e-3, batch_size=64) -> dict:
        M = X_train.shape[2]
        self._init_params = {"M": M, "n_classes": self.n_classes,
                             "hidden": self.hidden, "n_layers": self.n_layers, "dropout": self.dropout}
        self._build_model(**self._init_params)
        return self._train_loop(X_train, Y_train, X_val, Y_val, epochs, lr, batch_size)

    def _build_model(self, M=6, n_classes=2, hidden=64, n_layers=2, dropout=0.1, **_) -> None:
        self.model = _LSTMModel(M, n_classes, hidden, n_layers, dropout).to(self.device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._run_inference(X)
