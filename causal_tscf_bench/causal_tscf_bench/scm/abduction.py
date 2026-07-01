"""
Exact abduction — Plan Eq. 2.

U_t^(j) = X_t^(j) - sum_{X_{t-tau}^(i) in pa(X_t^(j))} w_ij * phi_ij(X_{t-tau}^(i))

Because mechanisms are additive and operators in INVERTIBLE_OPERATORS are
analytically computable (no neural network), this inversion is exact to
machine precision — no normalizing flow approximation is needed.

Reference:
  Pearl (2009), Causality (2nd ed.), Ch. 7 — abduction step of the causal ladder.
"""

from __future__ import annotations

import numpy as np

from .dag import LaggedDAG
from .operators import Mechanism


def abduct(
    X: np.ndarray,
    dag: LaggedDAG,
    mechanisms: list[Mechanism],
) -> np.ndarray:
    """
    Recover the exogenous noise sequence U from an observed trajectory X.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, T, M) or (T, M) for a single instance.
    dag : LaggedDAG
    mechanisms : list[Mechanism]

    Returns
    -------
    np.ndarray
        U with the same shape as X. Time steps t < max_lag have U = X (no parents
        reachable), which is correct: the burn-in window has no parent history.
    """
    single = X.ndim == 2
    if single:
        X = X[np.newaxis]   # (1, T, M)

    N, T, M = X.shape
    U = X.copy().astype(np.float64)

    mech_index: dict[tuple[int, int, int], Mechanism] = {
        (m.channel_to, m.channel_from, m.lag): m for m in mechanisms
    }

    for t in range(dag.max_lag, T):
        for j in range(M):
            parent_sum = np.zeros(N, dtype=np.float64)
            for (i, lag) in dag.parents_of(j):
                mech = mech_index.get((j, i, lag))
                if mech is None:
                    continue
                x_lagged = X[:, t - lag, i]
                parent_sum += mech.apply(x_lagged)
            U[:, t, j] = X[:, t, j] - parent_sum

    if single:
        return U[0]
    return U
