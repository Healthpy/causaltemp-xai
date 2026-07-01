"""
NlinearSCM-T benchmark — Week 4.

Nonlinear mechanisms sampled from the invertible operator dictionary Phi.
Exact abduction (Plan Eq. 2) remains valid because all operators in
INVERTIBLE_OPERATORS are analytically computable.

Config keys (same as LinearSCM-T plus):
  operator_pool : list[str] | null   subset of INVERTIBLE_OPERATORS keys
                                     (null => all seven invertible operators)
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkDataset, BenchmarkSplit
from ..scm.dag import sample_dag
from ..scm.operators import sample_mechanism, INVERTIBLE_OPERATORS
from ..scm.tscm import simulate_tscm, sample_noise


class NlinearSCMT(BenchmarkDataset):
    """NlinearSCM-T: nonlinear invertible-operator benchmark."""

    def generate(self) -> BenchmarkSplit:
        cfg = self.config
        rng = np.random.default_rng(self.seed)

        n_channels: int = cfg["n_channels"]
        max_lag: int = cfg.get("max_lag", 1)
        edge_density: float = cfg["edge_density"]
        noise_dist: str = cfg.get("noise_dist", "laplace")
        T: int = cfg["T"]
        N: int = cfg.get("N", 10_000)
        sigma_w: float = cfg.get("sigma_w", 1.0)
        burn_in: int = cfg.get("burn_in", 50)
        causal_ch: int | None = cfg.get("causal_channel", None)
        train_frac: float = cfg.get("train_frac", 0.70)
        val_frac: float = cfg.get("val_frac", 0.15)

        # Operator pool: default = all invertible; config can restrict to subset
        op_keys: list[str] | None = cfg.get("operator_pool", None)
        if op_keys is None:
            op_pool = INVERTIBLE_OPERATORS
        else:
            op_pool = {k: INVERTIBLE_OPERATORS[k] for k in op_keys if k in INVERTIBLE_OPERATORS}
            if not op_pool:
                raise ValueError(f"No valid invertible operators in {op_keys}")

        # Sample DAG and mechanisms
        dag = sample_dag(n_channels, max_lag, edge_density, rng)
        mechanisms = sample_mechanism(dag, rng, sigma_w=sigma_w, operator_pool=op_pool)

        # Scale weights to prevent explosion under nonlinear operators
        for m in mechanisms:
            m.weight = np.clip(m.weight, -1.5, 1.5)

        # Simulate
        noise = sample_noise(N, T + burn_in, n_channels, noise_dist, rng)
        X = simulate_tscm(dag, mechanisms, noise, burn_in=burn_in)

        # Clip diverged sequences (nonlinear operators can produce large values)
        X = np.clip(X, -50.0, 50.0)

        # Label generation
        if causal_ch is None:
            causal_ch = int(rng.integers(n_channels))
        Y, theta = self._label_from_threshold(X, causal_ch)

        # Split
        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        idx = rng.permutation(N)
        i_tr, i_va, i_te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

        ops_used = list({m.operator_key for m in mechanisms})
        meta = {
            "benchmark": "NlinearSCM-T",
            "n_channels": n_channels,
            "max_lag": max_lag,
            "edge_density": edge_density,
            "noise_dist": noise_dist,
            "T": T,
            "N": N,
            "sigma_w": sigma_w,
            "operator_pool": list(op_pool.keys()),
            "operators_used": ops_used,
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
