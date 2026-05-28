"""LinearSCM-T: Linear Structural Causal Model for Temporal data.

Generates multivariate time-series from a VAR(L) process defined by an
explicit causal graph.  Each variable x_i at time t is a linear function of
its causal parents at lags 1…L plus non-Gaussian noise:

    x_t  =  sum_{l=1}^{L}  A_l @ x_{t-l}  +  eps_t

where eps_t is drawn from either a Laplace or Uniform distribution.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


_NOISE_TYPES = ("laplace", "uniform")


class LinearSCMT:
    """VAR(L) data generator with a fixed causal structure.

    Parameters
    ----------
    k:
        Number of variables (dimensions) in the time series.
    L:
        Maximum lag of the VAR process.
    sparsity:
        Probability that any given lagged edge is present in the causal graph.
        Value in ``(0, 1]``; lower values produce sparser graphs.
    noise_type:
        Noise distribution for the innovation terms.  Either ``"laplace"`` or
        ``"uniform"``.
    T:
        Default trajectory length used by :meth:`generate`.
    N:
        Default number of samples used by :meth:`generate`.
    seed:
        Random seed for reproducibility.  Pass ``None`` for non-deterministic
        behaviour.
    """

    def __init__(
        self,
        k: int = 5,
        L: int = 2,
        sparsity: float = 0.3,
        noise_type: Literal["laplace", "uniform"] = "laplace",
        T: int = 50,
        N: int = 200,
        seed: Optional[int] = 42,
    ) -> None:
        if noise_type not in _NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {_NOISE_TYPES}, got {noise_type!r}")
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # Build the lagged graph and coefficient matrices once at construction
        self.graph, self.mechanisms = self._build_graph_and_mechanisms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, burn_in: int = 100) -> dict:
        """Generate a full dataset from the LinearSCM-T model.

        Parameters
        ----------
        burn_in:
            Number of initial time steps discarded to remove transient effects.

        Returns
        -------
        dict with keys:

        ``"X"``
            Float array of shape ``(N, T, k)`` – the observed time series.
        ``"Y"``
            Integer array of shape ``(N,)`` – binary labels derived from a
            threshold on the final-timestep latent value of variable 0.
        ``"graph"``
            Binary adjacency tensor of shape ``(k, k, L)``.  ``graph[i, j, l]``
            is 1 if variable *j* causes variable *i* at lag ``l+1``.
        ``"mechanisms"``
            List of *L* float arrays each of shape ``(k, k)`` – the VAR
            coefficient matrices (same ordering as ``graph``).
        """
        total_T = self.T + burn_in
        X_full = np.zeros((self.N, total_T, self.k))

        # Initialise first L steps with small noise
        for lag in range(self.L):
            X_full[:, lag, :] = self._sample_noise(self.N, self.k) * 0.1

        # Simulate VAR(L) forward
        noise = self._sample_noise(self.N * total_T * self.k).reshape(
            self.N, total_T, self.k
        )
        for t in range(self.L, total_T):
            x_t = noise[:, t, :].copy()
            for l, A in enumerate(self.mechanisms):
                x_t += X_full[:, t - l - 1, :] @ A.T
            X_full[:, t, :] = x_t

        X = X_full[:, burn_in:, :]  # shape (N, T, k)

        # Labels: threshold on final-timestep value of variable 0
        # Use the median across samples so classes are balanced by default
        latent_final = X[:, -1, 0]
        threshold = float(np.median(latent_final))
        Y = (latent_final > threshold).astype(int)

        return {
            "X": X,
            "Y": Y,
            "graph": self.graph,
            "mechanisms": self.mechanisms,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_graph_and_mechanisms(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Sample the causal graph and coefficient matrices.

        Returns
        -------
        graph : ndarray of shape ``(k, k, L)``
        mechanisms : list of L ndarrays each of shape ``(k, k)``
        """
        k, L = self.k, self.L
        graph = np.zeros((k, k, L), dtype=float)
        mechanisms = []

        for l in range(L):
            # Bernoulli mask for edges (no self-loops at lag 1)
            mask = (self._rng.random((k, k)) < self.sparsity).astype(float)
            if l == 0:
                np.fill_diagonal(mask, 0.0)
            graph[:, :, l] = mask

            # Draw coefficients and apply mask
            coefs = self._rng.uniform(-0.5, 0.5, (k, k))
            A = coefs * mask

            # Ensure stationarity: spectral radius < 0.9
            A = _stabilise(A, target_radius=0.9)
            mechanisms.append(A)

        return graph, mechanisms

    def _sample_noise(self, *shape) -> np.ndarray:
        """Draw noise samples from the configured non-Gaussian distribution."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        else:  # uniform
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1


def _stabilise(A: np.ndarray, target_radius: float = 0.9) -> np.ndarray:
    """Rescale *A* so its spectral radius is at most *target_radius*."""
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius > 1e-8:
        A = A * (target_radius / radius)
    return A
