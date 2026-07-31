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


def is_vacuous_intervention(
    x: np.ndarray,
    x_cf: np.ndarray,
    mechanism,
    t0: int | None = None,
    tol: float = INTERVENTION_TOL,
) -> bool:
    """Return True if ``x_cf`` differs from ``x`` but contains no ``do()``.

    A counterfactual claims that *something was set* at ``t0``. Under Pearl's
    action step an intervened value is one its parents do **not** determine, so
    a genuine intervention makes ``x_cf[t0]`` deviate from what the mechanism
    predicts given ``x_cf``'s own (unchanged) prefix. This returns True when
    that deviation is below ``tol`` — the CF's entire departure from the
    factual is mechanism continuation, not action.

    The case this exists for is the **noiseless-rollout family with a zero
    perturbation**. Such a CF drops the factual exogenous noise from ``t0``
    onward, so it differs from ``x`` (often by a lot — mean ``prox_l1 = 96.06``
    for CARLA on ``full_nl``) and clears every existing gate:
    ``derive_intervention_t`` finds a changed timestep, so the CF-faith
    degeneracy gate (``intervention_t >= T-1``) does not fire; the prefix is
    untouched, so the retroactive gate passes; and the CF *is* its own
    noiseless rollout, so the forward residual is zero and
    ``CFfaith(semantics="noiseless_rollout")`` scores a perfect
    ``hard=1.0, soft=1.0``. The reported proximity and sparsity then describe
    deleted noise rather than recourse.

    This is the pathology ``docs/risk_register.md`` RISK-16 charges the Bahri
    et al. (BigData 2025) causal-likelihood metric with — "a CF that is locally
    mechanism-consistent at every step while its intervention has no effect on
    the outcome scores perfectly". RISK-17 records that our own
    ``noiseless_rollout`` semantics shares the blind spot. Deliberately **not**
    folded into :class:`~causaltemp_xai.metrics.cf_faith.CFfaith`: CF-faith
    measures mechanism consistency and answers that question correctly here.
    Whether an intervention happened at all is a separate predicate, so it gets
    a separate function and leaves every published CF-faith number untouched
    (R5/R7 — no silent redefinition of a scored metric).

    Only meaningful against ``noiseless_rollout`` CFs. A ``pearl_delta`` CF
    with zero perturbation is a *literal* no-op (``x_cf == x``), which the
    CF-faith degeneracy gate already returns NaN for.

    Parameters
    ----------
    x, x_cf:
        Original and counterfactual trajectories, shape ``(T, k)``.
    mechanism:
        Mechanism supplying ``forward_numpy(window)`` — the deterministic
        next-step mean.
    t0:
        Claimed intervention timestep. Defaults to
        :func:`derive_intervention_t`, the benchmark's uniform heuristic.
    tol:
        Per-element threshold on ``|x_cf[t0] - mechanism(x_cf prefix)|``.
        Defaults to :data:`INTERVENTION_TOL`, the same "is this changed?"
        predicate ``derive_intervention_t`` and the CF-faith retroactive gate
        share (M1 decision, 2026-07-07).

    Returns
    -------
    bool
        True if the CF encodes no intervention. A literal no-op (``x_cf == x``)
        returns True — it intervened on nothing. ``t0 == 0`` returns False:
        there is no prefix to roll from, so an edit to the initial condition is
        always a genuine intervention.
    """
    from causaltemp_xai.benchmarks.mechanisms import lag_window

    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    k = x.shape[1]

    if t0 is None:
        t0 = derive_intervention_t(x, x_cf, tol=tol)

    # Literal no-op: nothing was set, trivially vacuous.
    if float(np.abs(x_cf - x).max()) <= tol:
        return True
    # No prefix to predict t0 from — an initial-condition edit is a real do().
    if t0 <= 0:
        return False

    predicted = mechanism.forward_numpy(lag_window(x_cf, t0, mechanism.L, k))
    return float(np.abs(x_cf[t0] - predicted).max()) <= tol


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
