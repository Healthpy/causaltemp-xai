"""
TSCM forward simulator.

Ported from causal_tscf_bench/causal_tscf_bench/scm/tscm.py.

X_t^(j) := sum_{X_{t-tau}^(i) in pa(X_t^(j))} w_ij * phi_ij(X_{t-tau}^(i)) + U_t^(j)

The simulator accepts pre-sampled exogenous noise U (shape: N x T x k) so that
abduction tests can verify round-trip accuracy by comparing recovered U to the
original U used here.

Variable naming: k = number of channels/variables (causaltemp_xai convention),
equivalent to M in the bench.

Reference:
  Runge et al. (2023), Causal inference for time series, Nat. Rev. Earth Env.
"""

from __future__ import annotations

import numpy as np

from .dag import LaggedDAG
from .operators import Mechanism


def simulate_tscm(
    dag: LaggedDAG,
    mechanisms: list[Mechanism],
    noise_seq: np.ndarray,
    burn_in: int = 50,
) -> np.ndarray:
    """
    Forward-simulate the TSCM given pre-sampled exogenous noise.

    Parameters
    ----------
    dag : LaggedDAG
    mechanisms : list[Mechanism]
        One Mechanism per edge as returned by sample_mechanism().
    noise_seq : np.ndarray
        Shape (N, T_total, k) where T_total = T + burn_in.
        U_t^(j) ~ noise distribution (Laplace or Uniform for identifiability).
    burn_in : int
        Number of initial time steps discarded to reduce dependence on zeros init.

    Returns
    -------
    np.ndarray
        Shape (N, T, k) — observed multivariate time series after burn-in.
    """
    N, T_total, k = noise_seq.shape
    T = T_total - burn_in

    # Build a lookup: mechanisms indexed by (channel_to, channel_from, lag)
    mech_index: dict[tuple[int, int, int], Mechanism] = {
        (m.channel_to, m.channel_from, m.lag): m for m in mechanisms
    }

    X = np.zeros((N, T_total + dag.max_lag, k), dtype=np.float64)
    # Pre-fill burn-in window with zeros (cold start)

    for t in range(dag.max_lag, T_total + dag.max_lag):
        noise_t = noise_seq[:, t - dag.max_lag, :]  # (N, k)
        X_t = noise_t.copy()
        for j in range(k):
            for (i, lag) in dag.parents_of(j):
                mech = mech_index.get((j, i, lag))
                if mech is None:
                    continue
                x_lagged = X[:, t - lag, i]   # (N,)
                contrib = mech.apply(x_lagged)
                contrib = np.nan_to_num(contrib, nan=0.0, posinf=50.0, neginf=-50.0)
                X_t[:, j] += contrib
        # Clip each time step to prevent divergence under nonlinear operators
        X_t = np.clip(X_t, -50.0, 50.0)
        X[:, t, :] = X_t

    # Discard max_lag padding and burn_in
    result = X[:, dag.max_lag + burn_in:, :]
    assert result.shape == (N, T, k), f"Shape mismatch: {result.shape}"
    return result


def sample_noise(
    N: int,
    T_total: int,
    k: int,
    distribution: str,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Sample exogenous noise sequences.

    Parameters
    ----------
    N : int
        Number of trajectories.
    T_total : int
        Total time steps (T + burn_in).
    k : int
        Number of channels/variables.
    distribution : str
        One of "laplace", "uniform", "gaussian".
        Non-Gaussian distributions ensure structural identifiability
        under the Darmois-Skitovic theorem (Hyvarinen & Morioka 2019).
    rng : np.random.Generator
    scale : float
        Scale parameter for the noise distribution.

    Returns
    -------
    np.ndarray
        Shape (N, T_total, k).
    """
    if distribution == "laplace":
        return rng.laplace(loc=0.0, scale=scale, size=(N, T_total, k))
    elif distribution == "uniform":
        return rng.uniform(-scale * np.sqrt(3), scale * np.sqrt(3), size=(N, T_total, k))
    elif distribution == "gaussian":
        # Gaussian ablation: breaks non-Gaussian identifiability assumption (used in H5)
        return rng.normal(0.0, scale, size=(N, T_total, k))
    else:
        raise ValueError(f"Unknown noise distribution: {distribution!r}")
