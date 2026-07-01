"""
Transformer classifier — patch-based encoder with CLS token.

Reference: Vaswani et al. (2017), Attention Is All You Need.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .base import TSClassifier


class _TransformerModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, d_model: int = 64,
                 nhead: int = 4, n_layers: int = 2, dropout: float = 0.1, max_T: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Embedding(max_T + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, T) -> (N, T, M)
        x = x.permute(0, 2, 1)
        N, T, _ = x.shape
        x = self.input_proj(x)   # (N, T, d_model)
        cls = self.cls_token.expand(N, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (N, T+1, d_model)
        pos_idx = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_idx)
        out = self.encoder(x)    # (N, T+1, d_model)
        return self.head(out[:, 0])   # CLS token


class TransformerClassifier(TSClassifier):
    def __init__(self, d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.dropout = dropout

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, epochs=50, lr=1e-3, batch_size=64) -> dict:
        M = X_train.shape[2]
        T = X_train.shape[1]
        self._init_params = {"M": M, "n_classes": self.n_classes, "d_model": self.d_model,
                             "nhead": self.nhead, "n_layers": self.n_layers,
                             "dropout": self.dropout, "max_T": T}
        self._build_model(**self._init_params)
        return self._train_loop(X_train, Y_train, X_val, Y_val, epochs, lr, batch_size)

    def _build_model(self, M=6, n_classes=2, d_model=64, nhead=4, n_layers=2,
                     dropout=0.1, max_T=512, **_) -> None:
        self.model = _TransformerModel(M, n_classes, d_model, nhead,
                                       n_layers, dropout, max_T=max_T).to(self.device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._run_inference(X)
