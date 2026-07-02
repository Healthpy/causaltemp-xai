"""
Intervention-time derivation and do-operator specification.

This module combines two sources:
1. derive_intervention_t() — MOVED from methods/intervention.py (causaltemp_xai
   original implementation). This is the benchmark's uniform intervention-time
   heuristic for CF methods that don't declare an explicit intervention point.

2. Intervention dataclass and apply_intervention() — ported from
   causal_tscf_bench/causal_tscf_bench/scm/intervention.py. These implement
   Pearl's do-operator (Action step of the causal ladder).

The derive_intervention_t() function is kept identical to the original in
methods/intervention.py so that existing callers in eval.py and cf_faith.py
continue to work after import path update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# MOVED from methods/intervention.py — causaltemp_xai original implementation
# ---------------------------------------------------------------------------


def derive_intervention_t(x: np.ndarray, x_cf: np.ndarray, tol: float = 1e-3) -> int:
    """Return the smallest ``t`` such that ``max_j |x_cf[t,j] - x[t,j]| > tol``.

    This is the benchmark's uniform heuristic: all CF methods (Wachter, DiCE,
    CARLA, cfts) are scored with the same intervention_t derived here so that
    CF-faith scores are comparable across methods.

    Parameters
    ----------
    x, x_cf:
        Original and counterfactual trajectories, shape ``(T, k)``.
    tol:
        Per-element threshold below which a timestep is considered unchanged.

    Returns
    -------
    int
        The first changed timestep. If no timestep differs (the CF equals the
        original within ``tol``), returns ``T - 1`` (degenerate; CF == original).
    """
    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    T = x.shape[0]
    # Max absolute deviation per timestep (over the feature axis/axes).
    per_t = np.abs(x_cf - x).reshape(T, -1).max(axis=1)
    changed = np.flatnonzero(per_t > tol)
    if changed.size == 0:
        return T - 1
    return int(changed[0])


# ---------------------------------------------------------------------------
# Ported from causal_tscf_bench — do-operator (Action step)
# ---------------------------------------------------------------------------


@dataclass
class Intervention:
    """Specification of a single do-operator intervention."""
    channel: int       # i — the intervened channel
    time: int          # T_int — the time step of intervention
    value: float       # x'_int — the constant value imposed

    def affects_time(self, t: int) -> bool:
        """True if time t is at or after the intervention time."""
        return t >= self.time


def apply_intervention(
    X_running: np.ndarray,
    t: int,
    j: int,
    intervention: "Intervention | None",
) -> np.ndarray:
    """
    During forward simulation, override X[t, j] if an intervention targets (j, t).

    Called inside compute_gt_counterfactual at each (t, j) step.
    Returns the (possibly overridden) value for channel j at time t.

    Parameters
    ----------
    X_running : np.ndarray
        Partially filled simulation array, shape (N, T_padded, k).
    t : int
        Current absolute time index in X_running.
    j : int
        Current channel being computed.
    intervention : Intervention | None

    Returns
    -------
    np.ndarray
        Shape (N,) — the value for X[:, t, j] after potential override.
    """
    value = X_running[:, t, j].copy()
    if intervention is not None and intervention.channel == j and intervention.time == t:
        value[:] = intervention.value
    return value
