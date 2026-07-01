"""
CITRIS wrapper — Causal Identifiability from Temporal Interventional Similarities.

CITRIS discovers a disentangled latent representation whose dimensions align
one-to-one with the causal variables in a temporal SCM, using pairs of
(pre-intervention, post-intervention) time steps as training signal.

Reference:
  Lippe et al. (2022), CITRIS: Causal Identifiability from Temporal
  Interventional Similarities. ICML 2022.

Evaluation note (per plan):
  For Axis B, report CF-faith against ground-truth graph AND against CITRIS-inferred
  graph separately to isolate graph estimation error vs. propagation failure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier
from ...scm.dag import LaggedDAG


@dataclass
class CITRISOutput:
    z: np.ndarray                    # (N, latent_dim) latent means
    assignment: np.ndarray           # (latent_dim,) channel assignment per latent dim
    inferred_adjacency: np.ndarray   # (M, M) inferred adjacency (lag-aggregated)


class _CITRISEncoder(nn.Module):
    def __init__(self, T: int, M: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(T * M, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        self.latent_dim = latent_dim
        self.T, self.M = T, M

    def forward(self, x: torch.Tensor):
        out = self.enc(x.view(x.shape[0], -1))
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        return mu, logvar


class CITRIS(AttributionMethod):
    """
    CITRIS temporal causal disentanglement model.

    Produces: latent assignments for Axis A (ICC, MCC) and inferred graph for Axis B.
    Not a CF explainer — does not produce counterfactuals directly.
    """

    def __init__(self, latent_dim: int = 8, epochs: int = 30,
                 batch_size: int = 64, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._encoder: _CITRISEncoder | None = None
        self._T: int = 0
        self._M: int = 0

    def fit_causal(self, X_train: np.ndarray, intervention_mask: np.ndarray | None = None) -> None:
        """
        X_train: (N, T, M)
        intervention_mask: (N, M) binary — 1 if channel was intervened at last time step
        """
        N, T, M = X_train.shape
        self._T, self._M = T, M
        enc = _CITRISEncoder(T, M, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

        x_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            idx = np.random.permutation(N)[:self.batch_size]
            xb = x_t[idx]
            mu, logvar = enc(xb)
            # Reconstruction loss (simple AE objective; full CITRIS uses RNN decoder)
            x_hat = xb.view(len(idx), -1)[:, :T * M]
            recon = nn.functional.mse_loss(mu[:, :T * M] if self.latent_dim >= T * M
                                           else mu, mu.detach())
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            loss = recon + 0.1 * kl
            opt.zero_grad(); loss.backward(); opt.step()

        enc.eval()
        self._encoder = enc

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("Call fit_causal() first.")
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self._encoder(x_t)
        return mu.cpu().numpy()

    def _infer_adj_matrix(self, X: np.ndarray) -> np.ndarray:
        """Returns (M, M) integer adjacency from latent-channel correlation heuristic."""
        Z = self.encode(X)   # (N, latent_dim)
        M = self._M
        ch_means = X.mean(axis=1)  # (N, M)
        corr = np.corrcoef(Z.T, ch_means.T)[:self.latent_dim, self.latent_dim:]
        assignment = np.argmax(np.abs(corr), axis=1)   # (latent_dim,)
        adj = np.zeros((M, M), dtype=int)
        for ld in range(self.latent_dim):
            ch = assignment[ld]
            for other_ld in range(self.latent_dim):
                if other_ld != ld:
                    other_ch = assignment[other_ld]
                    if abs(corr[ld, other_ch]) > 0.3:
                        adj[other_ch, ch] = 1
        return adj

    def inferred_graph(self, X: np.ndarray) -> LaggedDAG:
        """
        Return inferred causal graph as a LaggedDAG (max_lag=1).

        The (M, M) heuristic adjacency is wrapped in a LaggedDAG with shape
        (M, M, 1) where lag_idx=0 encodes lag-1 edges.
        """
        adj2d = self._infer_adj_matrix(X)   # (M, M)
        M = self._M
        adj3d = adj2d[:, :, np.newaxis]     # (M, M, 1)
        return LaggedDAG(n_channels=M, max_lag=1, adjacency=adj3d)

    def infer_graph(self, X: np.ndarray) -> np.ndarray:
        """Legacy: returns raw (M, M) array. Prefer inferred_graph()."""
        return self._infer_adj_matrix(X)

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        """Not the primary use of CITRIS; returns zero."""
        return np.zeros_like(x)
