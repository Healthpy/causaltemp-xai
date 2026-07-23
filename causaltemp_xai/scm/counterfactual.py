"""
Ground-truth analytical counterfactual via the three-step causal ladder.

Ported from causal_tscf_bench/causal_tscf_bench/scm/counterfactual.py.

Abduction  -> recover U from observed X
Action     -> apply do(X_{T_int}^(i) = x'_int)
Prediction -> forward-simulate with modified SCM and abduced U

The result X'_CF is the unique structural counterfactual that serves as the
absolute evaluation target for CF-faith (Plan Eq., Axis C).

REDUNDANCY NOTE: causaltemp_xai also has benchmarks/structural_cf.py which
implements the same three-step recipe using causaltemp_xai's LinearMechanism
(not the Mechanism dataclass). The two are functionally equivalent for the
linear case. This module uses the new Mechanism dataclass from scm/operators.py
and is the preferred path for new code using the scm/ package.

Reference:
  Pearl (2009), Causality (2nd ed.), Ch. 7.
  Schulam & Saria (2017), Reliable Decision Support using Counterfactual Models.
"""

from __future__ import annotations

import numpy as np

from .abduction import abduct
from .dag import LaggedDAG
from .intervention import Intervention
from .operators import Mechanism


def compute_gt_counterfactual(
    X: np.ndarray,
    dag: LaggedDAG,
    mechanisms: list[Mechanism],
    int_channel: int,
    int_time: int,
    int_value: float,
) -> np.ndarray:
    """
    Compute the analytical ground-truth counterfactual X'_CF.

    Parameters
    ----------
    X : np.ndarray
        Factual trajectory. Shape (N, T, k) or (T, k).
    dag : LaggedDAG
    mechanisms : list[Mechanism]
    int_channel : int
        Channel i targeted by the do-operator.
    int_time : int
        Time step T_int of the intervention (0-indexed within X).
    int_value : float
        Value x'_int imposed by the do-operator.

    Returns
    -------
    np.ndarray
        X'_CF with the same shape as X.
    """
    single = X.ndim == 2
    if single:
        X = X[np.newaxis]

    # Step 1 — Abduction
    U = abduct(X, dag, mechanisms)

    # Step 2 — Action (encode intervention)
    intervention = Intervention(channel=int_channel, time=int_time, value=int_value)

    # Step 3 — Prediction (forward simulate with abduced U and intervention)
    X_cf = _forward_simulate_with_intervention(X, U, dag, mechanisms, intervention)

    if single:
        return X_cf[0]
    return X_cf


def _forward_simulate_with_intervention(
    X_factual: np.ndarray,
    U: np.ndarray,
    dag: LaggedDAG,
    mechanisms: list[Mechanism],
    intervention: Intervention,
) -> np.ndarray:
    """
    Forward-simulate the SCM using abduced noise U and the do-intervention.

    Time steps before int_time are copied from X_factual (abduction guarantee:
    U encodes the exact background state, so pre-intervention values are
    recovered exactly). At int_time, channel int_channel is set to int_value.
    For t > int_time, all channels are re-simulated using the modified parent values.
    """
    N, T, k = X_factual.shape
    mech_index: dict[tuple[int, int, int], Mechanism] = {
        (m.channel_to, m.channel_from, m.lag): m for m in mechanisms
    }

    # Pad with dag.max_lag columns at the front using factual history
    pad = dag.max_lag
    X_cf_padded = np.zeros((N, T + pad, k), dtype=np.float64)
    # Fill pre-intervention history from factual (exact by abduction)
    X_cf_padded[:, :pad, :] = X_factual[:, :pad, :]

    for t_rel in range(T):
        t_abs = t_rel + pad  # index into padded array
        t_orig = t_rel  # index into X_factual / U

        for j in range(k):
            # Intervention override at the designated time step and channel
            if intervention.channel == j and t_orig == intervention.time:
                X_cf_padded[:, t_abs, j] = intervention.value
                continue

            # Structural equation: Eq. 1 with abduced noise
            parent_sum = np.zeros(N, dtype=np.float64)
            for i, lag in dag.parents_of(j):
                mech = mech_index.get((j, i, lag))
                if mech is None:
                    continue
                x_lagged = X_cf_padded[:, t_abs - lag, i]
                parent_sum += mech.apply(x_lagged)

            X_cf_padded[:, t_abs, j] = parent_sum + U[:, t_orig, j]

    return X_cf_padded[:, pad:, :]
