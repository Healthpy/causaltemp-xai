"""LinearSCMT benchmark with BenchmarkDataset base class.

MOVED from benchmark/generator.py and adapted to:
- Inherit from BenchmarkDataset (ABC from benchmarks/base.py)
- Return BenchmarkSplit from generate() instead of a plain dict
- Use causaltemp_xai 60/20/20 split ratio
- Keep all existing VAR(L) generation logic unchanged

NlinearSCMT is also included here for completeness.

The original dict-returning generate() is preserved as generate_dict()
for backward compatibility with code that accesses data["mechanism"] etc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from causaltemp_xai.benchmarks.base import BenchmarkDataset, BenchmarkSplit
from causaltemp_xai.benchmarks.mechanisms import LinearMechanism, MLPMechanism


_NOISE_TYPES = ("laplace", "uniform")


def _sample_lag_mask(k, lag_index, sparsity, rng):
    mask = (rng.random((k, k)) < sparsity).astype(float)
    if lag_index == 0:
        np.fill_diagonal(mask, 0.0)
    return mask


def _sample_graph(k, L, sparsity, rng):
    graph = np.zeros((k, k, L), dtype=float)
    for lag in range(L):
        graph[:, :, lag] = _sample_lag_mask(k, lag, sparsity, rng)
    return graph


def _stabilise(A, target_radius=0.9):
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius > 1e-8:
        A = A * (target_radius / radius)
    return A


class LinearSCMT(BenchmarkDataset):
    """VAR(L) data generator with a fixed causal structure.

    Now inherits from BenchmarkDataset. The generate() method returns
    BenchmarkSplit instead of a dict. Use generate_dict() for the old API.
    """

    def __init__(self, k=5, L=2, sparsity=0.3, noise_type="laplace",
                 T=50, N=200, seed=42):
        if noise_type not in _NOISE_TYPES:
            raise ValueError(
                "noise_type must be one of %s, got %r" % (_NOISE_TYPES, noise_type)
            )
        config = dict(k=k, L=L, sparsity=sparsity, noise_type=noise_type, T=T, N=N)
        super().__init__(config=config, seed=seed if seed is not None else 42)
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.graph, self.mechanism = self._build_graph_and_mechanisms()

    def generate(self, burn_in=100):
        """Generate a full dataset and return a BenchmarkSplit (60/20/20)."""
        d = self.generate_dict(burn_in=burn_in)
        X, Y = d["X"], d["Y"]
        train_idx, val_idx, test_idx = self._stratified_split(
            Y, seed=self.seed if self.seed is not None else 42
        )
        meta = dict(benchmark="LinearSCM-T", k=self.k, L=self.L,
                    sparsity=self.sparsity, noise_type=self.noise_type,
                    T=self.T, N=self.N, seed=self.seed,
                    split_fractions=[0.6, 0.2, 0.2])
        return BenchmarkSplit(
            X_train=X[train_idx], Y_train=Y[train_idx],
            X_val=X[val_idx], Y_val=Y[val_idx],
            X_test=X[test_idx], Y_test=Y[test_idx],
            graph=self.graph, mechanism=self.mechanism, meta=meta,
        )

    def generate_dict(self, burn_in=100):
        """Backward-compatible dict API: keys X, Y, graph, mechanism."""
        total_T = self.T + burn_in
        X_full = np.zeros((self.N, total_T, self.k))
        for lag in range(self.L):
            X_full[:, lag, :] = self._sample_noise(self.N, self.k) * 0.1
        noise = self._sample_noise(self.N * total_T * self.k).reshape(
            self.N, total_T, self.k)
        for t in range(self.L, total_T):
            window = X_full[:, t - self.L: t, :]
            x_t = noise[:, t, :].copy()
            x_t += self.mechanism.forward_numpy(window)
            X_full[:, t, :] = x_t
        X = X_full[:, burn_in:, :]
        Y = (X[:, -1, 0] > float(np.median(X[:, -1, 0]))).astype(int)
        return {"X": X, "Y": Y, "graph": self.graph, "mechanism": self.mechanism}

    def _build_graph_and_mechanisms(self):
        k, L = self.k, self.L
        graph = np.zeros((k, k, L), dtype=float)
        mechanisms = []
        for l in range(L):
            mask = _sample_lag_mask(k, l, self.sparsity, self._rng)
            graph[:, :, l] = mask
            coefs = self._rng.uniform(-0.5, 0.5, (k, k))
            A = _stabilise(coefs * mask, target_radius=0.9)
            mechanisms.append(A)
        return graph, LinearMechanism(mechanisms)

    def _sample_noise(self, *shape):
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        else:
            return self._rng.uniform(low=-0.17, high=0.17, size=size)


class NlinearSCMT(BenchmarkDataset):
    """Nonlinear SCM-T generator with BenchmarkDataset base class."""

    def __init__(self, k=5, L=2, sparsity=0.3, noise_type="laplace",
                 T=50, N=200, seed=42, hidden=16, gain=0.8,
                 decay_range=(0.3, 0.8), spectral_cap=0.9, init_gain=0.7,
                 activation="tanh", clip=1e3, max_resample=10):
        if noise_type not in _NOISE_TYPES:
            raise ValueError(
                "noise_type must be one of %s, got %r" % (_NOISE_TYPES, noise_type)
            )
        config = dict(k=k, L=L, sparsity=sparsity, noise_type=noise_type, T=T, N=N,
                      hidden=hidden, gain=gain, decay_range=list(decay_range),
                      spectral_cap=spectral_cap, init_gain=init_gain, activation=activation)
        super().__init__(config=config, seed=seed if seed is not None else 42)
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self.hidden = hidden
        self.gain = gain
        self.decay_range = decay_range
        self.spectral_cap = spectral_cap
        self.init_gain = init_gain
        self.activation = activation
        self.clip = clip
        self.max_resample = max_resample
        self._rng = np.random.default_rng(seed)
        self.graph = _sample_graph(self.k, self.L, self.sparsity, self._rng)
        self.mechanism = MLPMechanism.random(
            self.graph, hidden=self.hidden, rng=self._rng,
            decay_range=self.decay_range, gain=self.gain,
            spectral_cap=self.spectral_cap, init_gain=self.init_gain,
            activation=self.activation)

    def generate(self, burn_in=100):
        """Generate nonlinear dataset, return BenchmarkSplit (60/20/20)."""
        d = self.generate_dict(burn_in=burn_in)
        X, Y = d["X"], d["Y"]
        train_idx, val_idx, test_idx = self._stratified_split(
            Y, seed=self.seed if self.seed is not None else 42)
        meta = dict(benchmark="NlinearSCM-T", k=self.k, L=self.L,
                    sparsity=self.sparsity, noise_type=self.noise_type,
                    T=self.T, N=self.N, seed=self.seed,
                    split_fractions=[0.6, 0.2, 0.2])
        return BenchmarkSplit(
            X_train=X[train_idx], Y_train=Y[train_idx],
            X_val=X[val_idx], Y_val=Y[val_idx],
            X_test=X[test_idx], Y_test=Y[test_idx],
            graph=self.graph, mechanism=self.mechanism, meta=meta)

    def generate_dict(self, burn_in=100):
        """Backward-compatible dict API."""
        total_T = self.T + burn_in
        X_full = np.zeros((self.N, total_T, self.k))
        self._roll(X_full, np.arange(self.N), total_T)
        for _ in range(self.max_resample):
            bad = self._diverging(X_full)
            if not bad.any():
                break
            self._roll(X_full, np.nonzero(bad)[0], total_T)
        if self._diverging(X_full).any():
            np.clip(X_full, -self.clip, self.clip, out=X_full)
        X = X_full[:, burn_in:, :]
        Y = (X[:, -1, 0] > float(np.median(X[:, -1, 0]))).astype(int)
        return {"X": X, "Y": Y, "graph": self.graph, "mechanism": self.mechanism}

    def _roll(self, X_full, rows, total_T):
        rows = np.sort(np.asarray(rows))
        n = int(rows.size)
        if n == 0:
            return
        sub = np.zeros((n, total_T, self.k))
        for lag in range(self.L):
            sub[:, lag, :] = self._sample_noise(n, self.k) * 0.1
        noise = self._sample_noise(n * total_T * self.k).reshape(n, total_T, self.k)
        for t in range(self.L, total_T):
            window = sub[:, t - self.L: t, :]
            x_t = noise[:, t, :].copy()
            x_t += self.mechanism.forward_numpy(window)
            sub[:, t, :] = x_t
        X_full[rows] = sub

    def _diverging(self, X_full):
        finite = np.isfinite(X_full).all(axis=(1, 2))
        bounded = np.abs(np.nan_to_num(X_full, nan=np.inf)).max(axis=(1, 2)) <= self.clip
        return ~(finite & bounded)

    def _sample_noise(self, *shape):
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        else:
            return self._rng.uniform(low=-0.17, high=0.17, size=size)
