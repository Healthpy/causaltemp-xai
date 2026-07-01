"""
LinearSCM-T benchmark — Week 3.

VAR(1) synthetic benchmark with identity (linear) operators only.
Serves as the clean baseline for H1, H2, H4, H5 (Gaussian ablation).

Config keys (see configs/linear_scm_t.yaml):
  n_channels : int          k in {5, 10}
  max_lag    : int          L = 1
  edge_density : float      s/k^2 in {0.1, 0.2}
  noise_dist : str          "laplace" | "uniform" | "gaussian"
  T          : int          time-series length in {50, 100}
  N          : int          number of sequences (default 10000)
  sigma_w    : float        weight std (default 1.0; spectral radius clamped to <1)
  causal_channel : int | null   j* for label (null => random)
  train_frac : float        default 0.7
  val_frac   : float        default 0.15

Reference:
  Hauser & Buhlmann (2012), Characterization and Greedy Learning of
  Interventional Markov Equivalence Classes — motivates sparse DAG regime.
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkDataset, BenchmarkSplit
from ..scm.dag import sample_dag
from ..scm.operators import sample_mechanism, OPERATORS
from ..scm.tscm import simulate_tscm, sample_noise


class LinearSCMT(BenchmarkDataset):
    """LinearSCM-T: identity-operator VAR(L) synthetic benchmark."""

    def generate(self) -> BenchmarkSplit:
        cfg = self.config
        rng = np.random.default_rng(self.seed)

        n_channels: int = cfg["n_channels"]
        max_lag: int = cfg.get("max_lag", 1)
        edge_density: float = cfg["edge_density"]
        noise_dist: str = cfg.get("noise_dist", "laplace")
        T: int = cfg["T"]
        N: int = cfg.get("N", 10_000)
        sigma_w: float = cfg.get("sigma_w", 0.5)
        burn_in: int = cfg.get("burn_in", 50)
        causal_ch: int | None = cfg.get("causal_channel", None)
        train_frac: float = cfg.get("train_frac", 0.70)
        val_frac: float = cfg.get("val_frac", 0.15)

        # Sample DAG
        dag = sample_dag(n_channels, max_lag, edge_density, rng)

        # Linear mechanisms: identity operator only, weights clipped for stationarity
        identity_pool = {"identity": OPERATORS["identity"]}
        mechanisms = sample_mechanism(dag, rng, sigma_w=sigma_w, operator_pool=identity_pool)

        # Clamp weights to ensure approximate stationarity for VAR(1)
        for m in mechanisms:
            m.weight = np.clip(m.weight, -0.8 / n_channels, 0.8 / n_channels)

        # Sample noise and simulate
        noise = sample_noise(N, T + burn_in, n_channels, noise_dist, rng)
        X = simulate_tscm(dag, mechanisms, noise, burn_in=burn_in)

        # Label generation (Label Visibility)
        if causal_ch is None:
            causal_ch = int(rng.integers(n_channels))
        Y, theta = self._label_from_threshold(X, causal_ch)

        # Train / val / test split
        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        idx = rng.permutation(N)
        i_tr, i_va, i_te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

        meta = {
            "benchmark": "LinearSCM-T",
            "n_channels": n_channels,
            "max_lag": max_lag,
            "edge_density": edge_density,
            "noise_dist": noise_dist,
            "T": T,
            "N": N,
            "sigma_w": sigma_w,
            "causal_channel": causal_ch,
            "label_threshold": float(theta),
            "class_balance": float(Y.mean()),
            "seed": self.seed,
        }

        return BenchmarkSplit(
            X_train=X[i_tr], Y_train=Y[i_tr],
            X_val=X[i_va], Y_val=Y[i_va],
            X_test=X[i_te], Y_test=Y[i_te],
            dag=dag,
            mechanisms=mechanisms,
            meta=meta,
        )
