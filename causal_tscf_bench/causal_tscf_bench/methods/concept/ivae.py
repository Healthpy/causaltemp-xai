"""
iVAE wrapper — Identifiable Variational Autoencoder for disentangled latent concepts.

Used in Axis A (Latent Disentanglement) to measure whether latent dimensions
recover the ground-truth causal channels. iVAE leverages auxiliary segment
information (here: time-window index) to achieve identifiability.

Reference:
  Khemakhem et al. (2020), Variational Autoencoders and Nonlinear ICA:
  A Unifying Framework. AISTATS.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier


class _iVAEModel(nn.Module):
    def __init__(self, T: int, M: int, latent_dim: int, n_segments: int):
        super().__init__()
        input_dim = T * M
        # Encoder: q(z | x, u)
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim + n_segments, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim + n_segments, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Prior: p(z | u)
        self.prior_mu = nn.Linear(n_segments, latent_dim)
        self.prior_logvar = nn.Linear(n_segments, latent_dim)
        # Decoder: p(x | z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.T, self.M, self.latent_dim = T, M, latent_dim

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x_flat: torch.Tensor, u: torch.Tensor):
        xu = torch.cat([x_flat, u], dim=1)
        mu_q = self.encoder_mu(xu)
        lv_q = self.encoder_logvar(xu)
        z = self.reparametrize(mu_q, lv_q)
        x_hat = self.decoder(z)
        mu_p = self.prior_mu(u)
        lv_p = self.prior_logvar(u)
        return x_hat, mu_q, lv_q, mu_p, lv_p, z


class iVAE(AttributionMethod):
    """Identifiable VAE for causal disentanglement scoring."""

    def __init__(self, latent_dim: int = 8, n_segments: int = 4,
                 epochs: int = 20, batch_size: int = 64, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.n_segments = n_segments
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._model: _iVAEModel | None = None
        self._T: int = 0
        self._M: int = 0

    def fit_unsupervised(self, X_train: np.ndarray) -> None:
        N, T, M = X_train.shape
        self._T, self._M = T, M
        model = _iVAEModel(T, M, self.latent_dim, self.n_segments).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_flat = torch.tensor(X_train.reshape(N, -1), dtype=torch.float32)
        # Segment auxiliary: divide time series into n_segments windows
        seg_ids = np.arange(N) % self.n_segments
        u_one_hot = np.eye(self.n_segments)[seg_ids]
        u_t = torch.tensor(u_one_hot, dtype=torch.float32)

        dataset = TensorDataset(x_flat, u_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for xb, ub in loader:
                xb, ub = xb.to(self.device), ub.to(self.device)
                x_hat, mu_q, lv_q, mu_p, lv_p, z = model(xb, ub)
                recon = nn.functional.mse_loss(x_hat, xb)
                # KL(q(z|x,u) || p(z|u))
                kl = -0.5 * (1 + lv_q - lv_p
                             - (mu_q - mu_p).pow(2) / lv_p.exp()
                             - (lv_q - lv_p).exp()).sum(dim=1).mean()
                loss = recon + kl
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        self._model = model

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return latent means (N, latent_dim)."""
        if self._model is None:
            raise RuntimeError("Call fit_unsupervised() first.")
        N, T, M = X.shape
        x_flat = torch.tensor(X.reshape(N, -1), dtype=torch.float32, device=self.device)
        seg_ids = np.arange(N) % self.n_segments
        u = torch.tensor(np.eye(self.n_segments)[seg_ids], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            xu = torch.cat([x_flat, u], dim=1)
            mu = self._model.encoder_mu(xu)
        return mu.cpu().numpy()

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        """Not the primary use of iVAE — returns zero attribution by default."""
        return np.zeros((x.shape[0], x.shape[1]), dtype=np.float64)
