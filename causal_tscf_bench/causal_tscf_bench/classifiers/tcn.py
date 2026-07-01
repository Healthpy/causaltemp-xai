"""
TCN classifier — Temporal Convolutional Network with dilated residual blocks.

Reference:
  Bai et al. (2018), An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling. arXiv:1803.01271.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TSClassifier


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self._pad = (kernel - 1) * dilation   # left-only causal pad
        # No padding argument — we pad manually with F.pad to avoid right-side leakage
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad only: (pad_left, pad_right) = (self._pad, 0)
        # This ensures no future-context leakage (true causal convolution).
        out = self.relu(self.norm1(self.conv1(F.pad(x, (self._pad, 0)))))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(F.pad(out, (self._pad, 0)))))
        return self.relu(out + self.skip(x))


class _TCNModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 64, n_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = n_channels if i == 0 else hidden
            layers.append(_ResidualBlock(in_ch, hidden, kernel=3, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, T)
        out = self.tcn(x)           # (N, hidden, T)
        pooled = out[:, :, -1]      # last time step — label is a function of X_T
        return self.head(pooled)


class TCNClassifier(TSClassifier):
    def __init__(self, hidden: int = 64, n_layers: int = 6, dropout: float = 0.1, **kwargs):
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

    def _build_model(self, M=6, n_classes=2, hidden=64, n_layers=4, dropout=0.1, **_) -> None:
        self.model = _TCNModel(M, n_classes, hidden, n_layers, dropout).to(self.device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._run_inference(X)
