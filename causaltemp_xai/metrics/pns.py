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

Prior work: necessity/sufficiency for explanation is **not** new (LEWIS,
Galhotra et al. 2021; Watson et al. 2021). What is specific here is the
temporal setting with exact abduction plus the model-vs-world decomposition.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual
from causaltemp_xai.scm.intervention import INTERVENTION_TOL, derive_intervention_t

__all__ = [
    "extract_intervention",
    "pns_direction",
    "pns_from_directions",
    "recover_label_threshold",
    "scm_label",
]


def recover_label_threshold(X: np.ndarray, Y: np.ndarray) -> float:
    """Recover the generator's label threshold ``theta`` from a labelled split.

    The generators label by ``Y = 1[x[-1, 0] > theta]`` with
    ``theta = median(x[:, -1, 0])`` over all N samples, but **theta is not
    persisted** — ``generate()`` returns only ``X``/``Y``/``graph``/
    ``mechanism``. Without it the world-side outcome cannot be computed at all.

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
    v = X[:, -1, 0]
    theta = float(np.median(v))
    if not np.array_equal((v > theta).astype(int), Y):
        # Fall back to the label bracket: any value in [max{v:Y=0}, min{v:Y=1})
        # reproduces the labels. Used when the split is not the full generation
        # population, so the median is not recoverable exactly.
        lo, hi = v[Y == 0].max(), v[Y == 1].min()
        if not lo < hi:
            raise ValueError(
                "cannot recover label threshold: labels are not a clean "
                f"threshold on x[-1, 0] (max(v|Y=0)={lo!r} >= min(v|Y=1)={hi!r})"
            )
        theta = float((lo + hi) / 2)
        if not np.array_equal((v > theta).astype(int), Y):
            raise ValueError("recovered label threshold does not reproduce the given labels")
    return theta


def scm_label(x: np.ndarray, theta: float) -> int:
    """The generator's ground-truth label rule: ``1[x[-1, 0] > theta]``.

    This is the *world's* outcome — the data-generating process's own label,
    not the classifier's opinion of it.
    """
    return int(np.asarray(x, dtype=float)[-1, 0] > theta)


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


def pns_direction(
    X: np.ndarray,
    CFs: np.ndarray,
    model,
    mechanism,
    theta: float,
    target_class: int = 1,
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
    """
    X = np.asarray(X, dtype=float)
    CFs = np.asarray(CFs, dtype=float)
    if len(X) != len(CFs):
        raise ValueError(f"X ({len(X)}) and CFs ({len(CFs)}) batch sizes differ")

    predict = getattr(model, "predict", model)
    a_hits, b_hits, c_hits = [], [], []
    n_no_int = 0
    oracle_cfs = []
    keep = []

    for i, (x, x_cf) in enumerate(zip(X, CFs)):
        t0, nodes, values = extract_intervention(x, x_cf)
        if nodes.size == 0:
            n_no_int += 1
            continue
        # The world's realisation of *this* intervention: Pearl semantics.
        x_oracle = structural_counterfactual(x, mechanism, t0, nodes, values, noiseless=False)
        oracle_cfs.append(x_oracle)
        keep.append(i)
        c_hits.append(int(scm_label(x_oracle, theta) == target_class))

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
    }
