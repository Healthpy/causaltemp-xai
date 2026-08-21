"""Intervention-time derivation and the benchmark's shared "is this changed?" predicate.

:func:`derive_intervention_t` is the uniform intervention-time heuristic applied
to every CF method that does not declare an explicit intervention point, so
CF-faith, do-complexity and the PNS audit all score the *same* object rather
than three readings of one counterfactual. :data:`INTERVENTION_TOL` is the
single per-element threshold they share.

A ported ``Intervention`` dataclass and ``apply_intervention()`` also lived here
until 2026-08-03. Their only consumer was ``scm/counterfactual.py``, deleted the
same day as a duplicate of ``benchmarks/structural_cf.py``;
the do-operator's live implementation is that module's ``structural_counterfactual``
/ ``structural_counterfactual_schedule``.
"""

from __future__ import annotations

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

#: Scale-**relative** alternative: a per-channel threshold of this fraction of
#: the channel's own factual standard deviation (RISK-20, 2026-08-03).
#:
#: The absolute constant above is incoherent across channels, because the
#: channels are not on one scale: per-channel standard deviations differ by up
#: to **5.5x within a single config** (``smoke`` 0.144-0.791, ``full``
#: 0.178-0.579, ``full_nl`` 0.148-0.215). A fixed ``1e-3`` is therefore between
#: 0.13% and 0.69% of a channel's natural variation depending which channel it
#: lands on — so "changed" means something different per channel and per config.
#: That is what made ``CftsCels``'s do-complexity swing 1.0 -> 3.1 -> 13.2 across
#: three tolerance decades.
#:
#: 0.5% is chosen to sit in the *middle* of the absolute default's current
#: effective range, so adopting it is a recalibration rather than a tightening.
#: It is not tuned to any method's score.
#:
#: **Opt-in.** Every consumer defaults to ``rel_tol=None`` (absolute), so this
#: changes no committed number. Flipping the default is a separate decision
#: (2026-08-03).
RELATIVE_INTERVENTION_TOL = 0.005

#: Floor on a per-channel threshold, so a (near-)constant channel does not get a
#: zero tolerance and report float noise as an intervention.
_MIN_CHANNEL_TOL = 1e-12


def channel_tolerances(x: np.ndarray, rel_tol: float = RELATIVE_INTERVENTION_TOL) -> np.ndarray:
    """Per-channel "is this changed?" thresholds, scaled to the factual.

    Returns shape ``(k,)``: ``rel_tol * std(x[:, j])`` over the trajectory's own
    timesteps, floored at :data:`_MIN_CHANNEL_TOL`.

    Scaling to the **factual** ``x`` rather than the counterfactual is
    deliberate: the threshold must not move when the CF moves, or a method that
    edits more would grant itself a looser definition of "changed".
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(rel_tol * x.std(axis=0), _MIN_CHANNEL_TOL)


def _resolve_tol(x: np.ndarray, tol: float, rel_tol: float | None):
    """Scalar ``tol`` (absolute) or a ``(k,)`` array of per-channel thresholds.

    Both broadcast against a ``(k,)`` deviation row, so callers compare the same
    way in either mode.
    """
    return tol if rel_tol is None else channel_tolerances(x, rel_tol)


def derive_intervention_t(
    x: np.ndarray,
    x_cf: np.ndarray,
    tol: float = INTERVENTION_TOL,
    rel_tol: float | None = None,
) -> int:
    """Return the smallest ``t`` such that ``max_j |x_cf[t,j] - x[t,j]| > tol``.

    This is the benchmark's uniform heuristic: all CF methods (Wachter,
    NoiselessSCMRecourse, cfts) are scored with the same intervention_t derived here so that
    CF-faith scores are comparable across methods.

    Parameters
    ----------
    x, x_cf:
        Original and counterfactual trajectories, shape ``(T, k)``.
    tol:
        Absolute per-element threshold below which a timestep is considered
        unchanged (default :data:`INTERVENTION_TOL` — shared with the CF-faith
        retroactive gate; see the constant's docstring). Ignored when
        ``rel_tol`` is given.
    rel_tol:
        If given, use a **per-channel** threshold of ``rel_tol * std(x[:, j])``
        instead (:func:`channel_tolerances`, RISK-20). ``None`` (default)
        preserves the absolute behaviour byte-for-byte.

    Returns
    -------
    int
        The first changed timestep. If no timestep differs (the CF equals the
        original within tolerance), returns ``T - 1`` (degenerate; CF == original).
    """
    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    T = x.shape[0]
    dev = np.abs(x_cf - x).reshape(T, -1)
    if rel_tol is None:
        # Max absolute deviation per timestep (over the feature axis/axes).
        changed = np.flatnonzero(dev.max(axis=1) > tol)
    else:
        # Per-channel thresholds: a timestep is changed if ANY channel exceeds
        # its own threshold. Equivalent in form to the max-vs-scalar test, but
        # each channel is compared on its own scale.
        changed = np.flatnonzero((dev > channel_tolerances(x, rel_tol)).any(axis=1))
    if changed.size == 0:
        return T - 1
    return int(changed[0])


def is_vacuous_intervention(
    x: np.ndarray,
    x_cf: np.ndarray,
    mechanism,
    t0: int | None = None,
    tol: float = INTERVENTION_TOL,
    rel_tol: float | None = None,
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
    for NoiselessSCMRecourse on ``full_nl``) and clears every existing gate:
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

    thr = _resolve_tol(x, tol, rel_tol)
    if t0 is None:
        t0 = derive_intervention_t(x, x_cf, tol=tol, rel_tol=rel_tol)

    # Literal no-op: nothing was set, trivially vacuous.
    if bool(np.all(np.abs(x_cf - x) <= thr)):
        return True
    # No prefix to predict t0 from — an initial-condition edit is a real do().
    if t0 <= 0:
        return False

    predicted = mechanism.forward_numpy(lag_window(x_cf, t0, mechanism.L, k))
    return bool(np.all(np.abs(x_cf[t0] - predicted) <= thr))
