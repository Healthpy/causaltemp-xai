"""Batch evaluation pipeline for counterfactual methods.

Given a black-box model, a set of original instances, the counterfactuals a
method produced for them, and the SCM (graph + mechanisms), this computes the
full Axis-C metric suite plus **both** CF-faith semantics, averaged over the
batch.

The intervention timestep for each ``(x, x_cf)`` pair is derived uniformly via
:func:`~causaltemp_xai.methods.intervention.derive_intervention_t` (first
timestep where ``|x_cf - x| > tol``), so CF-faith is comparable across methods
that do not declare an intervention point themselves (e.g. Wachter).

Both CF-faith metrics are reported per the plan's "keep both CF-faith metrics"
decision: ``cf_faith_rollout_*`` (noiseless-rollout semantics, which CARLA is
built to satisfy) and ``cf_faith_pearl_*`` (Pearl delta-recursion). A single CF
cannot be ``hard=1`` under both; the contrast is itself a benchmark result.
"""

from __future__ import annotations

import inspect

import numpy as np

from causaltemp_xai.metrics.axis_c import (
    ood_plausibility,
    proximity,
    sparsity,
    validity,
)
from causaltemp_xai.metrics.cf_faith import CFfaith
from causaltemp_xai.scm.intervention import derive_intervention_t


def evaluate_method(
    model,
    X_orig: np.ndarray,
    CFs: np.ndarray,
    X_train: np.ndarray,
    graph: np.ndarray,
    mechanism,
    target_class: int = 1,
) -> dict:
    """Compute the batch-averaged metric suite for one CF method.

    Parameters
    ----------
    model:
        Classifier exposing ``predict`` (used for validity).
    X_orig:
        Original instances, shape ``(N, T, k)`` (a single ``(T, k)`` is
        promoted to a batch of 1).
    CFs:
        Counterfactuals produced for ``X_orig``, same shape and ordering.
    X_train:
        Training instances ``(M, T, k)`` used to fit the OOD detector.
    graph:
        SCM adjacency ``(k, k, L)``.
    mechanism:
        A :class:`~causaltemp_xai.benchmark.mechanisms.Mechanism` (the SCM
        transition), passed through to CF-faith.
    target_class:
        Desired output class for validity.

    Returns
    -------
    dict
        Flat dict of batch-mean metrics: ``validity``, ``proximity_l1``,
        ``proximity_l2``, ``sparsity``, ``frac_altered``,
        ``sparsity_channels`` / ``sparsity_timepoints`` (see below), ``ood``, the four
        CF-faith keys ``cf_faith_rollout_hard/soft`` and
        ``cf_faith_pearl_hard/soft``, and the two **joint
        faithfulness-validity** keys ``cf_faith_rollout_hard_valid`` /
        ``cf_faith_pearl_hard_valid`` — the fraction of instances that are
        *both* hard-faithful *and* classified as ``target_class``. The joint
        criterion closes the tiny-edit gameability loophole (M1, 2026-07-07):
        a near-no-op CF can pass the faithfulness check without consulting
        the SCM, but it then fails to flip the classifier, so it earns no
        joint credit. No proximity floor is needed — a CF that is tiny *and*
        valid *and* faithful is genuinely good, not gaming.  Also includes
        ``n`` (batch size).

        **Structured sparsity.** ``sparsity`` is a flat count over all ``T×k``
        features and so is blind to the *shape* of the edit. The two structured
        scores say which axis it is sparse along: ``sparsity_channels`` is the
        fraction of channels left entirely untouched across time, and
        ``sparsity_timepoints`` the fraction of timesteps left entirely
        untouched across channels. A method editing 1 of 5 channels at every
        timestep and one editing all 5 channels at a single timestep can post
        the *same* flat sparsity while being qualitatively different
        explanations — high ``sparsity_channels`` means "few variables", high
        ``sparsity_timepoints`` means "few moments". Both are in ``[0, 1]``,
        higher = sparser.
    """
    X_orig = np.asarray(X_orig, dtype=float)
    CFs = np.asarray(CFs, dtype=float)
    if X_orig.ndim == 2:
        X_orig = X_orig[np.newaxis]
    if CFs.ndim == 2:
        CFs = CFs[np.newaxis]
    if len(X_orig) != len(CFs):
        raise ValueError(f"X_orig ({len(X_orig)}) and CFs ({len(CFs)}) batch sizes differ")

    # Instantiate the two scorers once (outside the loop).
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")

    # Per-instance validity indicator (one predict call for the whole batch);
    # reused by both the validity mean and the joint criterion below.
    predict = getattr(model, "predict", model)
    preds = np.asarray(predict(CFs)).reshape(-1)
    valid_i = (preds == target_class).astype(float)  # (N,)

    prox_l1, prox_l2, spars = [], [], []
    spars_ch, spars_tp = [], []
    r_hard, r_soft, p_hard, p_soft = [], [], [], []

    for x, x_cf in zip(X_orig, CFs):
        prox_l1.append(proximity(x, x_cf, norm="l1"))
        prox_l2.append(proximity(x, x_cf, norm="l2"))
        spars.append(sparsity(x, x_cf))
        # Structured sparsity: *which axis* the edit is sparse along. The
        # flat score above cannot tell a few-channels-all-timesteps edit from
        # a few-timesteps-all-channels one; these two can.
        detailed = sparsity(x, x_cf, return_detailed=True)
        spars_ch.append(detailed["channels"])
        spars_tp.append(detailed["timepoints"])

        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mechanism)
        p = pearl.score(x, x_cf, t, graph, mechanism)
        r_hard.append(r["hard"])
        r_soft.append(r["soft"])
        p_hard.append(p["hard"])
        p_soft.append(p["soft"])

    ood_scores = np.atleast_1d(ood_plausibility(X_train, CFs))
    sparsity_mean = float(np.mean(spars))

    return {
        "n": len(CFs),
        "validity": float(np.mean(valid_i)),
        "proximity_l1": float(np.mean(prox_l1)),
        "proximity_l2": float(np.mean(prox_l2)),
        "sparsity": sparsity_mean,
        "frac_altered": float(1.0 - sparsity_mean),
        "sparsity_channels": float(np.mean(spars_ch)),
        "sparsity_timepoints": float(np.mean(spars_tp)),
        "ood": float(np.mean(ood_scores)),
        "cf_faith_rollout_hard": float(np.mean(r_hard)),
        "cf_faith_rollout_soft": float(np.mean(r_soft)),
        "cf_faith_pearl_hard": float(np.mean(p_hard)),
        "cf_faith_pearl_soft": float(np.mean(p_soft)),
        # Joint faithfulness-validity criterion (anti-gameability, M1):
        # fraction of instances that are BOTH hard-faithful AND valid.
        "cf_faith_rollout_hard_valid": float(np.mean(np.asarray(r_hard) * valid_i)),
        "cf_faith_pearl_hard_valid": float(np.mean(np.asarray(p_hard) * valid_i)),
    }


#: Minimum base validity below which the Shift-VR *ratio* is not reported
#: (M1 decision, 2026-07-07). A retention ratio with a small denominator is
#: numerically unstable at benchmark sample sizes: at ``n_cf=10``,
#: ``validity_base=0.3`` means 3 successes, and a swing of a few instances
#: moves the ratio by integer multiples (the CftsConfeti 2.67x artifact).
#: Below this floor only the ``(validity_base, validity_shift)`` pair is
#: reported; the ratio is ``None``.
MIN_VALIDITY_BASE_FOR_RATIO = 0.3


def _generate_batch(method, X, model, graph, mechanism):
    """Call ``method.generate_batch`` with the right signature.

    CARLA-style recourse needs ``graph``/``mechanism``; Wachter/cfts-* do not.
    We inspect the signature rather than special-casing class names.
    """
    params = inspect.signature(method.generate_batch).parameters
    if "graph" in params or "mechanism" in params:
        return method.generate_batch(X, model, graph, mechanism)
    return method.generate_batch(X, model)


def shift_vr(
    model,
    methods: dict,
    X_base_test: np.ndarray,
    X_shift_test: np.ndarray,
    graph: np.ndarray,
    mechanism,
    target_class: int = 1,
    cf_base: dict | None = None,
) -> dict:
    """Shift-VR-lite validity-retention metric (Axis D).

    Protocol (pinned — see ``resources/configs.md``): keep the **frozen base
    classifier** (no retraining). For each method, generate *fresh* CFs for the
    base test inputs and for the shifted test inputs, measure validity on each,
    and report the retention ratio ``validity(E_shift) / validity(E_base)``.

    The signal comes from generating recourse for shifted inputs — **not** from
    re-checking a fixed CF array, whose validity cannot change under a single
    frozen classifier.

    Parameters
    ----------
    model:
        The frozen base classifier (exposing ``predict`` / ``torch_logits``).
    methods:
        Mapping ``{name: cf_method}``; each value exposes ``generate_batch``
        (Wachter/cfts-*: ``(X, model)``; CARLA: ``(X, model, graph, mechanism)``).
    X_base_test, X_shift_test:
        Test inputs ``(N, T, k)`` from the base and shifted environments.
    graph, mechanism:
        The (shared) SCM structure, passed to causal methods.
    target_class:
        Desired output class for validity.
    cf_base:
        Optional ``{name: (N, T, k)}`` of **already-generated** CFs for
        ``X_base_test``. Any method found here skips base regeneration; the
        rest fall back to generating.

        This is both a large speedup and a **correctness fix** (2026-07-15).
        Regenerating the base CFs duplicated work the caller had already done
        — Phase 03 generates exactly these arrays, persists them, and Phase 04
        scores them — so the phase paid for every method's recourse twice.
        Worse, for a **stochastic** method the throwaway regeneration is a
        *different* CF array than the persisted one: ``CftsCounts`` is provably
        nondeterministic (two regenerations on identical inputs give validity
        0.40 and 0.30 where the persisted array is 0.60), so ``validity_base``
        described a counterfactual set that nothing else in the pipeline ever
        saw and that disagreed with the validity Phase 04 reports for the same
        method+config. Passing the persisted arrays makes the Shift-VR
        denominator *the same object* the rest of the pipeline scores.

        Note the ratio's numerator (``validity_shift``) is still a fresh
        generation by construction — that is the metric's actual signal — so
        Shift-VR remains noisy for stochastic methods; the
        ``MIN_VALIDITY_BASE_FOR_RATIO`` guard and multi-seed CIs are what
        control that, not this parameter.

    Returns
    -------
    dict
        ``{name: {"validity_base", "validity_shift", "shift_vr", "n_base",
        "n_shift"}}``. The ``(validity_base, validity_shift)`` pair is always
        reported. ``shift_vr`` is ``validity_shift / validity_base`` **only
        when** ``validity_base >= MIN_VALIDITY_BASE_FOR_RATIO`` (0.3);
        otherwise it is ``None`` — a retention ratio on a tiny denominator is
        a small-sample artifact, not a robustness signal (M1 guard,
        2026-07-07; ``None`` also serializes to valid JSON ``null``, unlike
        the previous ``nan`` sentinel).
    """
    results: dict = {}
    cf_base = cf_base or {}
    for name, method in methods.items():
        # Reuse caller-supplied base CFs when available. Generating recourse is
        # by far the dominant cost here, and a caller that already ran this
        # method on X_base_test (Phase 03 does, then saves the array) would
        # otherwise pay for the identical computation twice: _generate_batch
        # is deterministic given (method, X, model, graph, mechanism), so the
        # regenerated array is the one the caller already holds.
        base_cfs = cf_base.get(name)
        if base_cfs is None:
            base_cfs = _generate_batch(method, X_base_test, model, graph, mechanism)
        cf_shift = _generate_batch(method, X_shift_test, model, graph, mechanism)
        v_base = validity(base_cfs, model, target_class)
        v_shift = validity(cf_shift, model, target_class)
        if v_base >= MIN_VALIDITY_BASE_FOR_RATIO:
            ratio = v_shift / v_base
        else:
            ratio = None  # pair still reported; ratio would be unstable
        results[name] = {
            "validity_base": v_base,
            "validity_shift": v_shift,
            "shift_vr": ratio,
            "n_base": int(np.asarray(X_base_test).shape[0]),
            "n_shift": int(np.asarray(X_shift_test).shape[0]),
        }
    return results
