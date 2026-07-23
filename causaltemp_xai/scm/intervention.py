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

#: Single source of truth for "is this element changed?" across the benchmark.
#:
#: ``derive_intervention_t`` (which *defines* the intervention timestep t0)
#: and the ``CFfaith`` retroactive gate share this
#: per-element threshold (M1 decision, 2026-07-07). Rationale: t0 is defined
#: as the first timestep whose **per-element max** deviation exceeds this
#: tolerance, so any downstream retroactive check must use the *same
#: predicate at the same scale* — otherwise a region certified "unchanged" by
#: the definition of t0 can be flagged "retroactively edited" by its own
#: consumer (the CELS false-flag artifact: summed 1e-4 vs per-element 1e-3).
#: A per-element max is also T×k-invariant: a summed check accumulates
#: floating-point reconstruction noise linearly in the pre-window area.
INTERVENTION_TOL = 1e-3


def derive_intervention_t(x: np.ndarray, x_cf: np.ndarray, tol: float = INTERVENTION_TOL) -> int:
    """Return the smallest ``t`` such that ``max_j |x_cf[t,j] - x[t,j]| > tol``.

    This is the benchmark's uniform heuristic: all CF methods (Wachter,
    CARLA, cfts) are scored with the same intervention_t derived here so that
    CF-faith scores are comparable across methods.

    Parameters
    ----------
    x, x_cf:
        Original and counterfactual trajectories, shape ``(T, k)``.
    tol:
        Per-element threshold below which a timestep is considered unchanged
        (default :data:`INTERVENTION_TOL` — shared with the CF-faith
        retroactive gate; see the constant's docstring).

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

    channel: int  # i — the intervened channel
    time: int  # T_int — the time step of intervention
    value: float  # x'_int — the constant value imposed

    def affects_time(self, t: int) -> bool:
        """True if time t is at or after the intervention time."""
        return t >= self.time


def apply_intervention(
    X_running: np.ndarray,
    t: int,
    j: int,
    intervention: Intervention | None,
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
