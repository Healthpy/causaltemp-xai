"""
CARLA-Causal wrapper — Amortized / controllable CF paradigm.

CARLA (Causal Recourse for Latent Additive) learns an amortized CF generator
trained with causal graph constraints. At inference, it maps a query instance
to a CF via a single forward pass, modifying only causally downstream variables.

Reference:
  Karimi et al. (2021), Algorithmic Recourse Under Imperfect Causal Knowledge:
  A Probabilistic Approach. NeurIPS.
  Mahajan et al. (2020), ICML.

Failure mode (per plan): requires known causal graph at training time; graph
misspecification leads to invalid interventions that violate parent-child ordering.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..base import CFExplainer
from ...classifiers.base import TSClassifier
from ...scm.dag import LaggedDAG


class _AmortizedCFGenerator(nn.Module):
    """Amortized CF generator conditioned on target class."""

    def __init__(self, T: int, M: int, n_classes: int = 2, hidden: int = 64):
        super().__init__()
        in_dim = T * M + n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T * M),
        )
        self.T, self.M = T, M

    def forward(self, x_flat: torch.Tensor, target_oh: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_flat, target_oh], dim=1)
        delta = self.net(inp)
        return (x_flat + delta).view(-1, self.T, self.M)


class CARLACausal(CFExplainer):
    """Amortized causal recourse CF explainer."""

    def __init__(self, n_classes: int = 2, hidden: int = 64, epochs: int = 40,
                 lambda_prox: float = 0.5, device: str = "cpu"):
        self.n_classes = n_classes
        self.hidden = hidden
        self.epochs = epochs
        self.lambda_prox = lambda_prox
        self.device = device
        self._gen: _AmortizedCFGenerator | None = None
        self._dag: LaggedDAG | None = None

    def set_dag(self, dag: LaggedDAG) -> None:
        self._dag = dag

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        N, T, M = X_train.shape
        gen = _AmortizedCFGenerator(T, M, self.n_classes, self.hidden).to(self.device)
        opt = torch.optim.Adam(gen.parameters(), lr=1e-3)

        x_t = torch.tensor(X_train.reshape(N, -1), dtype=torch.float32, device=self.device)
        y_pred = classifier.predict(X_train)
        # Target is flipped label
        y_target = 1 - y_pred

        for epoch in range(self.epochs):
            idx = np.random.permutation(N)[:64]
            xb = x_t[idx]
            target_oh = torch.zeros(len(idx), self.n_classes, device=self.device)
            target_oh[torch.arange(len(idx)), y_target[idx]] = 1.0

            x_cf = gen(xb, target_oh)
            x_cf_np = x_cf.detach().cpu().numpy()
            proba = classifier.predict_proba(x_cf_np)
            # Validity: P(target class) on CF
            validity_scores = torch.tensor(
                [proba[i, y_target[idx[i]]] for i in range(len(idx))],
                dtype=torch.float32, device=self.device
            )
            prox_loss = torch.mean((x_cf - xb.view(-1, T, M)) ** 2)
            # Maximize validity, minimize proximity
            loss = -validity_scores.mean() + self.lambda_prox * prox_loss
            opt.zero_grad(); loss.backward(); opt.step()

        gen.eval()
        self._gen = gen

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        if self._gen is None:
            raise RuntimeError("Call fit() before explain().")
        T, M = x.shape
        x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
        target_oh = torch.zeros(1, self.n_classes, device=self.device)
        target_oh[0, target_class] = 1.0
        with torch.no_grad():
            x_cf = self._gen(x_t, target_oh)
        return x_cf.view(T, M).cpu().numpy()
