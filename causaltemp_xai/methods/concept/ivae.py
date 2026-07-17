"""
iVAE wrapper — Identifiable Variational Autoencoder for disentangled latent concepts.

Used in Axis A (Latent Disentanglement) to measure whether latent dimensions
recover the ground-truth causal channels. iVAE leverages auxiliary segment
information (here: time-window index) to achieve identifiability.

Reference:
  Khemakhem et al. (2020), Variational Autoencoders and Nonlinear ICA:
  A Unifying Framework. AISTATS.

Ported from causal_tscf_bench/methods/concept/ivae.py.
Note: fit_unsupervised uses segment-conditioned prior for identifiability.
The attribute() method is a no-op; use encode() + metrics.axis_a for scoring.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ...classifiers.base import TSClassifier
from ..base import AttributionMethod


class _iVAEModel(nn.Module):
    def __init__(self, T: int, k: int, latent_dim: int, n_segments: int):
        super().__init__()
        input_dim = T * k
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim + n_segments, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim + n_segments, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Segment-conditioned prior — required for identifiability guarantee
        self.prior_mu = nn.Linear(n_segments, latent_dim)
        self.prior_logvar = nn.Linear(n_segments, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.T, self.k, self.latent_dim = T, k, latent_dim

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
    """Identifiable VAE for causal disentanglement scoring (Axis A).

    Usage
    -----
    1. Call fit_unsupervised(X_train) to train the segment-conditioned VAE.
    2. Call encode(X) to get latent means (N, latent_dim) for MCC/LD scoring.
    3. attribute() returns zeros — use encode() + metrics.axis_a instead.
    """

    def __init__(self, latent_dim: int = 8, n_segments: int = 4,
                 epochs: int = 20, batch_size: int = 64, device: str = "cpu",
                 beta: float = 1.0, kl_warmup_frac: float = 0.3):
        self.latent_dim = latent_dim
        self.n_segments = n_segments
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.beta = beta
        # KL warmup: linearly ramp the KL weight from 0 to ``beta`` over the
        # first ``kl_warmup_frac`` of training, to avoid posterior collapse
        # (the decoder otherwise ignores z and outputs the data mean).
        self.kl_warmup_frac = kl_warmup_frac
        self._model: _iVAEModel | None = None
        self._T: int = 0
        self._k: int = 0

    def fit_unsupervised(self, X_train: np.ndarray) -> None:
        """Train the segment-conditioned iVAE on unlabelled time series.

        Parameters
        ----------
        X_train : (N, T, k)
        """
        N, T, k = X_train.shape
        self._T, self._k = T, k
        model = _iVAEModel(T, k, self.latent_dim, self.n_segments).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_flat = torch.tensor(X_train.reshape(N, -1), dtype=torch.float32)
        seg_ids = np.arange(N) % self.n_segments
        u_one_hot = np.eye(self.n_segments)[seg_ids]
        u_t = torch.tensor(u_one_hot, dtype=torch.float32)

        loader = DataLoader(TensorDataset(x_flat, u_t),
                            batch_size=self.batch_size, shuffle=True)

        warmup_epochs = max(1, int(self.kl_warmup_frac * self.epochs))
        model.train()
        for ep in range(self.epochs):
            kl_weight = self.beta * min(1.0, (ep + 1) / warmup_epochs)
            for xb, ub in loader:
                xb, ub = xb.to(self.device), ub.to(self.device)
                x_hat, mu_q, lv_q, mu_p, lv_p, _ = model(xb, ub)
                recon = nn.functional.mse_loss(x_hat, xb)
                kl = -0.5 * (
                    1 + lv_q - lv_p
                    - (mu_q - mu_p).pow(2) / lv_p.exp()
                    - (lv_q - lv_p).exp()
                ).sum(dim=1).mean()
                loss = recon + kl_weight * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        self._model = model

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return posterior means Z of shape (N, latent_dim)."""
        if self._model is None:
            raise RuntimeError("Call fit_unsupervised() first.")
        N = X.shape[0]
        x_flat = torch.tensor(X.reshape(N, -1), dtype=torch.float32,
                               device=self.device)
        seg_ids = np.arange(N) % self.n_segments
        u = torch.tensor(np.eye(self.n_segments)[seg_ids], dtype=torch.float32,
                         device=self.device)
        with torch.no_grad():
            xu = torch.cat([x_flat, u], dim=1)
            mu = self._model.encoder_mu(xu)
        return mu.cpu().numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Deterministically decode latents ``Z`` ``(N, latent_dim)`` back to
        time series ``(N, T, k)`` (the decoder path only — no reparam sampling).

        Used by :func:`causaltemp_xai.metrics.axis_a.icc_latent` for the
        latent-traversal ICC.
        """
        if self._model is None:
            raise RuntimeError("Call fit_unsupervised() first.")
        Z_t = torch.tensor(np.asarray(Z), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x_hat = self._model.decoder(Z_t)
        return x_hat.cpu().numpy().reshape(-1, self._T, self._k)

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        """iVAE is not an attribution method — returns zeros.
        Use encode() + metrics.axis_a.latent_disentanglement() instead.
        """
        return np.zeros((x.shape[0], x.shape[1]), dtype=np.float64)
