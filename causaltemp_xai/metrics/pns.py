"""Necessity/sufficiency gap: the *model's* causal claim vs the *world's*.

CF-faith (``metrics/cf_faith.py``) asks a **trajectory** question — is this CF
consistent with the mechanism? It cannot ask whether the intervention the CF
proposes actually *causes* the outcome change it claims. The two come apart in
both directions, and the benchmark's own results show each:

* CARLA on ``full``: CF-faith(rollout) = 1.00 and validity = 1.00, yet the
  intervention's effect on the final timestep is ~2.6e-04. The label flip comes
  from the noiseless rollout discarding the abducted noise, not from the
  intervention (2026-07-30 diagnosis, ``DECISIONS.md``). This module is the
  metric that catches that.
* CARLA on ``full_nl``: CF-faith = 1.00, validity = 0.00 — perfectly faithful,
  causally inert.

**Exact, not bounded.** Necessity/sufficiency probabilities are normally only
*bounded* (Tian & Pearl 2000) because both potential outcomes are never jointly
observed. Under this benchmark's additive-noise SCMs abduction is an exact
subtraction, so for a given instance both potential outcomes are deterministic
and computable — the estimand is exact. That is the main reason this metric is
worth defining here specifically.

The three quantities
--------------------
For a factual ``x``, a method's counterfactual ``x_cf``, and the intervention
implied by ``x_cf`` at the benchmark's uniform ``derive_intervention_t``
heuristic (the same intervention CF-faith scores, so the two axes describe one
object):

=========  =================  ==============================  ==========================================
symbol     outcome function   trajectory                      reads as
=========  =================  ==============================  ==========================================
``A``      classifier ``f``   the method's ``x_cf``           what the explanation claims
``B``      classifier ``f``   oracle CF of *that* intervention claim, evaluated on the world's trajectory
``C``      SCM label rule     oracle CF of that intervention  what the world actually does
=========  =================  ==============================  ==========================================

The oracle CF uses **Pearl semantics** (reuse the abducted factual noise): the
question is what would have happened *to this instance*, not what the noiseless
skeleton does.

Decomposition
-------------
::

    delta_total      = A - C
    delta_trajectory = A - B    (proposed CF != what the world would produce)
    delta_outcome    = B - C    (classifier != the true label mechanism)
    delta_total      = delta_trajectory + delta_outcome   (exact, by construction)

Signed, and both directions carry meaning: **> 0** the explanation claims a
causal efficacy the world denies (spurious / model artifact); **< 0** the model
is blind to a real causal dependence.

PN, PS and the naming
---------------------
On the standard flip-candidate set every instance already has
``f(x) != target``, so ``Y_{x'} = 0`` holds *by construction* and
``P(Y_x=1, Y_{x'}=0)`` collapses to ``P(Y_x=1)`` — exactly validity. The
estimand there is the **probability of sufficiency**, not PNS. Genuine PNS
therefore needs both directions, combined via

    PNS = P(x,y)·PN + P(x',y')·PS

so :func:`pns_from_directions` takes both and reports the terms *separately* as
well as combined — the combined number alone can hide a collapsed PN (R3
naming honesty; see ``docs/pns_metric_design.md``).

How the intervention is read: do-complexity
-------------------------------------------
``A``/``B``/``C`` are only as meaningful as the intervention extracted from
``x_cf``, and there is no single correct extraction. Reading the CF as one
``do()`` at ``t0`` (:func:`extract_intervention`, the default and what every
committed number uses) audits a densely-editing method against an intervention
it never proposed, inflating ``delta_trajectory``. Reading *every* edited slice
as a ``do()`` makes the oracle reproduce ``x_cf`` exactly, so
``delta_trajectory == 0`` by construction. Both endpoints are tautologies in
opposite directions, so the benchmark reports the bracket:
:func:`do_complexity` counts how many timesteps a proposal must declare as
actions before the mechanism can produce it, and is published beside
``delta_trajectory`` in both modes (RISK-18, ``ROADMAP.md`` M2b).

Prior work: necessity/sufficiency for explanation is **not** new (LEWIS,
Galhotra et al. 2021; Watson et al. 2021). What is specific here is the
temporal setting with exact abduction plus the model-vs-world decomposition.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.benchmarks.labels import get_label_functional
from causaltemp_xai.benchmarks.mechanisms import lag_window
from causaltemp_xai.benchmarks.structural_cf import (
    abduct_noise,
    structural_counterfactual,
    structural_counterfactual_schedule,
)
from causaltemp_xai.scm.intervention import INTERVENTION_TOL, derive_intervention_t

__all__ = [
    "do_complexity",
    "extract_intervention",
    "extract_intervention_schedule",
    "pns_direction",
    "pns_from_directions",
    "recover_label_threshold",
    "scm_label",
]


def _resolve_label_fn(label_fn):
    """Accept a :class:`LabelFunctional`, a registry name, or ``None``."""
    if label_fn is None or isinstance(label_fn, str):
        return get_label_functional(label_fn)
    return label_fn


def recover_label_threshold(X: np.ndarray, Y: np.ndarray, label_fn=None) -> float:
    """Recover the generator's label threshold ``theta`` from a labelled split.

    The generators label by ``Y = 1[g(x) > theta]`` with ``theta`` the median of
    ``g`` over all N samples, where ``g`` is the config's label functional
    (``benchmarks/labels.py``; ``label_fn=None`` is the default terminal rule
    ``g(x) = x[-1, 0]``). **theta is not persisted** — ``generate()`` returns
    only ``X``/``Y``/``graph``/``mechanism``. Without it the world-side outcome
    cannot be computed at all.

    Pass the **same** functional the config generated with. Passing the wrong
    one does not silently corrupt the result: ``g`` will not reproduce the
    stored labels and this raises.

    Recomputing the median over the *concatenated* splits recovers it exactly
    (verified 2026-07-30 on ``full``, ``full_nl``, ``smoke``, ``smoke_nl``,
    ``smoke_gaussian``, ``smoke_nonmonotonic``, ``smoke_regime``: the recovered
    value reproduces every stored label). Pass the concatenation of
    train+val+test, not a single split — the median of one split is not the
    median of the whole.

    A wrong ``theta`` silently corrupts every world-side number, so this
    validates itself and raises rather than returning a value that disagrees
    with the labels it was derived from.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).astype(int).reshape(-1)
    v = _resolve_label_fn(label_fn).latent_batch(X)
    theta = float(np.median(v))
    if not np.array_equal((v > theta).astype(int), Y):
        # Fall back to the label bracket: any value in [max{v:Y=0}, min{v:Y=1})
        # reproduces the labels. Used when the split is not the full generation
        # population, so the median is not recoverable exactly.
        lo, hi = v[Y == 0].max(), v[Y == 1].min()
        if not lo < hi:
            raise ValueError(
                "cannot recover label threshold: labels are not a clean "
                f"threshold on the label functional (max(v|Y=0)={lo!r} >= "
                f"min(v|Y=1)={hi!r}). If this config was generated with a "
                "non-default label_fn, pass the same one."
            )
        theta = float((lo + hi) / 2)
        if not np.array_equal((v > theta).astype(int), Y):
            raise ValueError("recovered label threshold does not reproduce the given labels")
    return theta


def scm_label(x: np.ndarray, theta: float, label_fn=None) -> int:
    """The generator's ground-truth label rule: ``1[g(x) > theta]``.

    This is the *world's* outcome — the data-generating process's own label,
    not the classifier's opinion of it. ``g`` is the config's label functional;
    ``label_fn=None`` is the default terminal rule ``x[-1, 0]``, which is what
    every committed number was computed with.

    **Which ``g`` is in force is not a detail (RISK-19).** While ``g`` reads the
    terminal timestep, the label site coincides with the trajectory end, so the
    horizon claim cannot be told apart from a claim about terminal labelling.
    ``benchmarks/labels.py`` exists to break that tie.
    """
    return int(_resolve_label_fn(label_fn).latent_one(x) > theta)


def extract_intervention(x: np.ndarray, x_cf: np.ndarray, tol: float = INTERVENTION_TOL):
    """``(t0, nodes, values)`` for the intervention a CF implies.

    ``t0`` comes from :func:`derive_intervention_t` — the benchmark's uniform
    heuristic, reused so PNS and CF-faith score the *same* intervention rather
    than two different readings of one CF. ``nodes`` are the channels the CF
    actually changed at ``t0`` (per-element, same ``tol`` as the t0 definition,
    so a channel certified unchanged there cannot appear here), and ``values``
    are what it set them to.

    Returns ``nodes`` empty when the CF changes nothing at ``t0`` — callers
    must treat that as "no intervention" rather than as an intervention with
    no effect.
    """
    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    t0 = derive_intervention_t(x, x_cf, tol=tol)
    changed = np.flatnonzero(np.abs(x_cf[t0] - x[t0]) > tol)
    return t0, changed, x_cf[t0, changed]


def extract_intervention_schedule(
    x: np.ndarray,
    x_cf: np.ndarray,
    mechanism,
    tol: float = INTERVENTION_TOL,
    semantics: str = "pearl_delta",
):
    """Every timestep ``x_cf`` sets by **action** rather than by continuation.

    Returns a list of ``(t, nodes, values)`` in increasing ``t``, suitable for
    :func:`~causaltemp_xai.benchmarks.structural_cf.structural_counterfactual_schedule`.

    **What this fixes (RISK-18).** :func:`extract_intervention` reads only the
    first deviating slice, so a proposal that edits many timesteps is audited
    against a one-slice ``do()`` it never made, and the resulting
    ``delta_trajectory`` cannot be told apart from "this method edited more than
    one timestep". Walking the whole trajectory recovers what the proposal
    actually asserts.

    The predicate is :func:`~causaltemp_xai.scm.intervention.is_vacuous_intervention`'s,
    applied at every ``t`` instead of only at ``t0``: a coordinate is an *action*
    iff it deviates from what the mechanism would produce from ``x_cf``'s own
    prefix. ``is_vacuous_intervention(x, x_cf, mech)`` is therefore the special
    case "is the schedule empty at ``t0``", and ``frac_vacuous`` is the ``D = 0``
    row of the do-complexity distribution.

    Parameters
    ----------
    x, x_cf:
        Factual and counterfactual trajectories, shape ``(T, k)``.
    mechanism:
        Mechanism supplying ``forward_numpy(window)``.
    tol:
        Per-element "is this changed?" threshold — :data:`INTERVENTION_TOL`, the
        same predicate that defines ``t0`` and gates CF-faith's retroactive
        check, so a coordinate certified unchanged there cannot appear here.
    semantics:
        Which continuation the proposal is read against, mirroring
        :class:`~causaltemp_xai.metrics.cf_faith.CFfaith`'s two semantics.
        ``"pearl_delta"`` (default) re-injects the noise abducted from ``x``, and
        matches the oracle :func:`pns_direction` scores against.
        ``"noiseless_rollout"`` uses the deterministic skeleton. The two give
        very different answers for the noiseless-rollout method family, and that
        is informative rather than a defect: a CF that discards the factual noise
        is not reproducible by a Pearl oracle from *any* small set of
        interventions, so its ``pearl_delta`` do-complexity is near-maximal —
        which is the quantitative form of the RISK-17 vacuity finding.

    Returns
    -------
    list of ``(t, nodes, values)`` — empty if ``x_cf`` asserts no action at all
    (a literal no-op, or a pure mechanism continuation of the factual prefix).
    """
    if semantics not in ("pearl_delta", "noiseless_rollout"):
        raise ValueError(
            f"unknown semantics {semantics!r}; expected 'pearl_delta' or 'noiseless_rollout'"
        )
    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    T, k = x.shape
    L = mechanism.L

    if float(np.abs(x_cf - x).max()) <= tol:
        return []

    eps = (
        abduct_noise(x, mechanism) if semantics == "pearl_delta" else np.zeros((T, k), dtype=float)
    )

    t0 = derive_intervention_t(x, x_cf, tol=tol)
    schedule = []
    for t in range(t0, T):
        if t == 0:
            # No prefix to predict from, so an edit to the initial condition is
            # always a genuine do() — matching is_vacuous_intervention's t0 == 0
            # branch rather than reading the zero-padded window as a prediction.
            changed = np.flatnonzero(np.abs(x_cf[t] - x[t]) > tol)
        else:
            predicted = mechanism.forward_numpy(lag_window(x_cf, t, L, k)) + eps[t]
            changed = np.flatnonzero(np.abs(x_cf[t] - predicted) > tol)
        if changed.size:
            schedule.append((t, changed, x_cf[t, changed]))
    return schedule


def do_complexity(
    x: np.ndarray,
    x_cf: np.ndarray,
    mechanism,
    tol: float = INTERVENTION_TOL,
    semantics: str = "pearl_delta",
) -> int:
    """``D`` — how many timesteps ``x_cf`` must declare as ``do()`` to be realisable.

    ``D = len(extract_intervention_schedule(...))``. Report it **beside**
    ``delta_trajectory``, never instead of it: the two endpoints of the reading
    are both tautological on their own. Scoring a proposal as a single-slice
    ``do()`` inflates ``delta_trajectory`` for any method that edits densely
    (RISK-18); scoring it as "every edited slice is an intervention" makes the
    oracle reproduce ``x_cf`` exactly, so ``delta_trajectory == 0`` **by
    construction**. The informative statement is the pair: a causally coherent CF
    sets a few values and lets the mechanism produce the rest (``D`` small, and
    ``delta_trajectory`` small in schedule mode *because* the world agrees); a
    direct edit of the outcome's neighbourhood buys the same
    ``delta_trajectory`` only by declaring most of the trajectory to be
    intervened on, which ``D`` makes visible.

    ``D = 0`` is exactly the vacuous case (RISK-17) — the CF differs from the
    factual but asserts no action anywhere.
    """
    return len(extract_intervention_schedule(x, x_cf, mechanism, tol=tol, semantics=semantics))


def pns_direction(
    X: np.ndarray,
    CFs: np.ndarray,
    model,
    mechanism,
    theta: float,
    target_class: int = 1,
    schedule: bool = False,
    label_fn=None,
) -> dict:
    """Score one direction (PS *or* PN) of the model-vs-world gap.

    Which one it is depends entirely on the instances passed: the standard
    flip-candidate set yields **PS**, its complement (instances already in
    ``target_class``, with CFs seeking to leave it) yields **PN**. This
    function does not know or care which — :func:`pns_from_directions`
    combines them.

    Returns the three quantities ``A``/``B``/``C`` of the module docstring,
    the additive decomposition, and the diagnostics needed to read them
    honestly (``n_scorable``, ``frac_no_intervention``).

    Instances where the CF implies **no intervention at all** are excluded, not
    scored 0: as with CF-faith's degeneracy gate, a no-op carries no evidence
    about causal efficacy and must abstain rather than dilute the mean.

    Parameters
    ----------
    schedule:
        ``False`` (default) reads each CF as a single ``do()`` at ``t0``, which
        is what every committed number was computed with — the default is
        load-bearing and must stay bit-identical (R7). ``True`` reads the full
        multi-timestep schedule the CF implies
        (:func:`extract_intervention_schedule`) and realises all of it. Neither
        reading is "the" answer; see :func:`do_complexity`. ``D`` is reported in
        **both** modes, so a single-slice run still exposes how much of the
        proposal it is declining to model.
    label_fn:
        The world's label rule (``benchmarks/labels.py``). Must be the one the
        config generated with, and the same one ``theta`` was recovered under —
        ``C`` is meaningless otherwise. ``None`` is the default terminal rule.
    """
    X = np.asarray(X, dtype=float)
    CFs = np.asarray(CFs, dtype=float)
    label = _resolve_label_fn(label_fn)
    if len(X) != len(CFs):
        raise ValueError(f"X ({len(X)}) and CFs ({len(CFs)}) batch sizes differ")

    predict = getattr(model, "predict", model)
    a_hits, b_hits, c_hits = [], [], []
    n_no_int = 0
    oracle_cfs = []
    keep = []
    d_values = []

    for i, (x, x_cf) in enumerate(zip(X, CFs)):
        sched = extract_intervention_schedule(x, x_cf, mechanism)
        if schedule:
            if not sched:
                n_no_int += 1
                continue
            # The world's realisation of *the whole proposal*: Pearl semantics.
            x_oracle = structural_counterfactual_schedule(x, mechanism, sched, noiseless=False)
        else:
            t0, nodes, values = extract_intervention(x, x_cf)
            if nodes.size == 0:
                n_no_int += 1
                continue
            # The world's realisation of *this* intervention: Pearl semantics.
            x_oracle = structural_counterfactual(x, mechanism, t0, nodes, values, noiseless=False)
        d_values.append(len(sched))
        oracle_cfs.append(x_oracle)
        keep.append(i)
        c_hits.append(int(scm_label(x_oracle, theta, label_fn=label) == target_class))

    n_scorable = len(keep)
    if n_scorable == 0:
        nan = float("nan")
        return {
            "n": len(X),
            "n_scorable": 0,
            "frac_no_intervention": 1.0 if len(X) else nan,
            "A_model_proposed": nan,
            "B_model_oracle": nan,
            "C_world_oracle": nan,
            "delta_total": nan,
            "delta_trajectory": nan,
            "delta_outcome": nan,
            "do_complexity_mean": nan,
            "do_complexity_median": nan,
            "schedule_mode": bool(schedule),
        }

    # Batch the two classifier calls rather than one per instance.
    a_hits = (np.asarray(predict(CFs[keep])).reshape(-1) == target_class).astype(float)
    b_hits = (np.asarray(predict(np.stack(oracle_cfs))).reshape(-1) == target_class).astype(float)
    c_hits = np.asarray(c_hits, dtype=float)

    A = float(a_hits.mean())
    B = float(b_hits.mean())
    C = float(c_hits.mean())
    return {
        "n": len(X),
        "n_scorable": n_scorable,
        "frac_no_intervention": float(n_no_int / len(X)),
        "A_model_proposed": A,
        "B_model_oracle": B,
        "C_world_oracle": C,
        "delta_total": A - C,
        "delta_trajectory": A - B,
        "delta_outcome": B - C,
        "do_complexity_mean": float(np.mean(d_values)),
        "do_complexity_median": float(np.median(d_values)),
        "schedule_mode": bool(schedule),
    }


def pns_from_directions(ps_dir: dict, pn_dir: dict, p_xy: float, p_xpyp: float) -> dict:
    """Combine the two directions into PNS via ``PNS = P(x,y)·PN + P(x',y')·PS``.

    ``ps_dir`` / ``pn_dir`` are :func:`pns_direction` outputs for the
    sufficiency and necessity directions; ``p_xy`` and ``p_xpyp`` are the
    population weights ``P(X=x, Y=y)`` and ``P(X=x', Y=y')``.

    **The combined number is never reported alone.** PN can collapse toward 0
    for structural reasons — an intervention placed many steps before the
    outcome cannot move it in *either* direction (the 2026-07-30 horizon
    finding) — and a combined PNS would then quietly report the PS term while
    looking like a joint quantity. PN, PS and the weights are all returned
    alongside so a collapsed term is visible.
    """
    ps = ps_dir["C_world_oracle"]
    pn = pn_dir["C_world_oracle"]
    pns = p_xy * pn + p_xpyp * ps
    return {
        "PNS_world": float(pns),
        "PS_world": float(ps),
        "PN_world": float(pn),
        "p_xy": float(p_xy),
        "p_xpyp": float(p_xpyp),
        "PS_delta_total": ps_dir["delta_total"],
        "PN_delta_total": pn_dir["delta_total"],
        "PS_n_scorable": ps_dir["n_scorable"],
        "PN_n_scorable": pn_dir["n_scorable"],
        # Carried through for the same reason PN and PS are: a PNS computed
        # from proposals with near-maximal do-complexity is not measuring the
        # same object as one computed from sparse, genuinely causal edits.
        "PS_do_complexity_mean": ps_dir.get("do_complexity_mean", float("nan")),
        "PN_do_complexity_mean": pn_dir.get("do_complexity_mean", float("nan")),
    }
