"""
NlinearSCM-T ablations — Week 6.

Two sub-configurations that stress-test specific aspects of H6 and H7:

NlinearSCMT_Nonmonotonic
  Adds the step function to the operator pool, violating monotonicity.
  Exact abduction (Eq. 2) is no longer valid for step-function edges;
  GT CFs for these edges use an approximate noise estimate.
  Used to test H6: generative flow models (DoFlow, Glacier) fail here.

NlinearSCMT_Regime
  HMM with R regimes; active weights w_ij switch at structural break-points.
  Regime identity is stored in split.meta["regime_labels"].
  Used to test H7: CoMTE Shift-VR degrades faster than CITRIS.

Reference:
  Hamilton (1989), A New Approach to the Economic Analysis of Nonstationary
  Time Series — HMM regime-switching model.
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkDataset, BenchmarkSplit
from ..scm.dag import sample_dag
from ..scm.operators import sample_mechanism, OPERATORS, INVERTIBLE_OPERATORS
from ..scm.tscm import simulate_tscm, sample_noise


class NlinearSCMT_Nonmonotonic(BenchmarkDataset):
    """
    NlinearSCM-T with step functions in the operator pool.

    Note: exact abduction is undefined for step-function edges.
    GT CFs revert to approximate noise (U ≈ 0) for those edges.
    The meta dict flags `identifiable: false`.
    """

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

        # Full operator pool including non-invertible step
        op_pool = OPERATORS

        dag = sample_dag(n_channels, max_lag, edge_density, rng)
        mechanisms = sample_mechanism(dag, rng, sigma_w=sigma_w, operator_pool=op_pool)
        for m in mechanisms:
            m.weight = np.clip(m.weight, -1.5, 1.5)

        noise = sample_noise(N, T + burn_in, n_channels, noise_dist, rng)
        X = simulate_tscm(dag, mechanisms, noise, burn_in=burn_in)
        X = np.clip(X, -50.0, 50.0)

        if causal_ch is None:
            causal_ch = int(rng.integers(n_channels))
        Y, theta = self._label_from_threshold(X, causal_ch)

        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        idx = rng.permutation(N)
        i_tr, i_va, i_te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

        step_edges = sum(1 for m in mechanisms if m.operator_key == "step")
        meta = {
            "benchmark": "NlinearSCM-T-Nonmonotonic",
            "identifiable": False,
            "step_edges": step_edges,
            "n_channels": n_channels,
            "T": T,
            "N": N,
            "noise_dist": noise_dist,
            "causal_channel": causal_ch,
            "label_threshold": float(theta),
            "class_balance": float(Y.mean()),
            "seed": self.seed,
        }

        return BenchmarkSplit(
            X_train=X[i_tr], Y_train=Y[i_tr],
            X_val=X[i_va], Y_val=Y[i_va],
            X_test=X[i_te], Y_test=Y[i_te],
            dag=dag, mechanisms=mechanisms, meta=meta,
        )


class NlinearSCMT_Regime(BenchmarkDataset):
    """
    NlinearSCM-T with HMM regime-switching.

    A hidden Markov process with n_regimes states triggers structural breaks
    that modify the active weights w_ij at random change-points, producing
    controlled non-stationarity for Shift-VR evaluation (H7).
    """

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
        n_regimes: int = cfg.get("n_regimes", 2)
        transition_prob: float = cfg.get("transition_prob", 0.05)
        min_regime_len: int = cfg.get("min_regime_len", 15)
        train_frac: float = cfg.get("train_frac", 0.70)
        val_frac: float = cfg.get("val_frac", 0.15)

        dag = sample_dag(n_channels, max_lag, edge_density, rng)

        # Sample one mechanism set per regime
        regime_mechanisms = [
            sample_mechanism(dag, rng, sigma_w=sigma_w, operator_pool=INVERTIBLE_OPERATORS)
            for _ in range(n_regimes)
        ]
        for mech_list in regime_mechanisms:
            for m in mech_list:
                m.weight = np.clip(m.weight, -1.2, 1.2)

        # Simulate regime sequence using HMM transitions
        X_all = []
        regime_labels_all = []

        for _ in range(N):
            regime_seq = _sample_regime_sequence(
                T + burn_in, n_regimes, transition_prob, min_regime_len, rng
            )
            noise_single = sample_noise(1, T + burn_in, n_channels, noise_dist, rng)

            # Simulate step-by-step switching mechanisms per time step
            x_seq = _simulate_regime_switching(
                dag, regime_mechanisms, noise_single[0], regime_seq, burn_in
            )
            X_all.append(x_seq)
            regime_labels_all.append(regime_seq[burn_in:])  # post burn-in

        X = np.stack(X_all, axis=0)   # (N, T, M)
        X = np.clip(X, -50.0, 50.0)
        regime_labels = np.stack(regime_labels_all, axis=0)   # (N, T)

        if causal_ch is None:
            causal_ch = int(rng.integers(n_channels))
        Y, theta = self._label_from_threshold(X, causal_ch)

        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        idx = rng.permutation(N)
        i_tr, i_va, i_te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

        meta = {
            "benchmark": "NlinearSCM-T-Regime",
            "n_regimes": n_regimes,
            "transition_prob": transition_prob,
            "min_regime_len": min_regime_len,
            "n_channels": n_channels,
            "T": T,
            "N": N,
            "noise_dist": noise_dist,
            "causal_channel": causal_ch,
            "label_threshold": float(theta),
            "class_balance": float(Y.mean()),
            "seed": self.seed,
        }

        split = BenchmarkSplit(
            X_train=X[i_tr], Y_train=Y[i_tr],
            X_val=X[i_va], Y_val=Y[i_va],
            X_test=X[i_te], Y_test=Y[i_te],
            dag=dag,
            mechanisms=regime_mechanisms[0],  # regime 0 as reference for GT CFs
            meta=meta,
        )
        # Attach regime labels for Axis D Shift-VR computation
        split.meta["regime_labels_train"] = regime_labels[i_tr].tolist()
        split.meta["regime_labels_test"] = regime_labels[i_te].tolist()
        return split


def _sample_regime_sequence(
    T: int,
    n_regimes: int,
    transition_prob: float,
    min_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a hidden Markov regime label sequence of length T."""
    regimes = np.zeros(T, dtype=np.int32)
    current = rng.integers(n_regimes)
    since_last_switch = 0
    for t in range(T):
        regimes[t] = current
        since_last_switch += 1
        if since_last_switch >= min_len and rng.random() < transition_prob:
            candidates = [r for r in range(n_regimes) if r != current]
            current = int(rng.choice(candidates))
            since_last_switch = 0
    return regimes


def _simulate_regime_switching(
    dag,
    regime_mechanisms: list,
    noise: np.ndarray,   # (T_total, M)
    regime_seq: np.ndarray,   # (T_total,)
    burn_in: int,
) -> np.ndarray:
    """Simulate TSCM with regime-dependent mechanisms, step by step."""
    from ..scm.operators import Mechanism as M_cls

    T_total, n_channels = noise.shape
    pad = dag.max_lag
    X = np.zeros((T_total + pad, n_channels), dtype=np.float64)

    # Build per-regime mechanism lookup
    regime_mech_index = []
    for mech_list in regime_mechanisms:
        idx = {(m.channel_to, m.channel_from, m.lag): m for m in mech_list}
        regime_mech_index.append(idx)

    for t in range(pad, T_total + pad):
        t_noise = t - pad
        regime = int(regime_seq[t_noise])
        mech_index = regime_mech_index[regime]
        for j in range(n_channels):
            parent_sum = 0.0
            for (i, lag) in dag.parents_of(j):
                mech = mech_index.get((j, i, lag))
                if mech:
                    contrib = float(mech.apply(X[t - lag, i]))
                    if np.isfinite(contrib):
                        parent_sum += contrib
            val = parent_sum + noise[t_noise, j]
            X[t, j] = np.clip(val, -50.0, 50.0)

    return X[pad + burn_in:]
