"""Batch evaluation pipeline for counterfactual methods.

Given a black-box model, a set of original instances, the counterfactuals a
method produced for them, and the SCM (graph + mechanisms), this computes the
full Axis-C metric suite plus **both** CF-faith semantics, averaged over the
batch.

The intervention timestep for each ``(x, x_cf)`` pair is derived uniformly via
:func:`~causaltemp_xai.methods.intervention.derive_intervention_t` (first
timestep where ``|x_cf - x| > tol``), so CF-faith is comparable across methods
that do not declare an intervention point themselves (Wachter, DiCE).

Both CF-faith metrics are reported per the plan's "keep both CF-faith metrics"
decision: ``cf_faith_rollout_*`` (noiseless-rollout semantics, which CARLA is
built to satisfy) and ``cf_faith_pearl_*`` (Pearl delta-recursion). A single CF
cannot be ``hard=1`` under both; the contrast is itself a benchmark result.
"""

from __future__ import annotations

import inspect

import numpy as np

from causaltemp_xai.methods.intervention import derive_intervention_t
from causaltemp_xai.metrics.axis_c import (
    ood_plausibility,
    proximity,
    sparsity,
    validity,
)
from causaltemp_xai.metrics.cf_faith import CFfaith


def evaluate_method(
    model,
    X_orig: np.ndarray,
    CFs: np.ndarray,
    X_train: np.ndarray,
    graph: np.ndarray,
    mechanisms: list,
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
    mechanisms:
        List of ``L`` VAR coefficient matrices ``A_l``, each ``(k, k)``.
    target_class:
        Desired output class for validity.

    Returns
    -------
    dict
        Flat dict of batch-mean metrics: ``validity``, ``proximity_l1``,
        ``proximity_l2``, ``sparsity``, ``frac_altered``, ``ood``, and the four
        CF-faith keys ``cf_faith_rollout_hard/soft`` and
        ``cf_faith_pearl_hard/soft``.  Also includes ``n`` (batch size).
    """
    X_orig = np.asarray(X_orig, dtype=float)
    CFs = np.asarray(CFs, dtype=float)
    if X_orig.ndim == 2:
        X_orig = X_orig[np.newaxis]
    if CFs.ndim == 2:
        CFs = CFs[np.newaxis]
    if len(X_orig) != len(CFs):
        raise ValueError(
            f"X_orig ({len(X_orig)}) and CFs ({len(CFs)}) batch sizes differ"
        )

    # Instantiate the two scorers once (outside the loop).
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")

    prox_l1, prox_l2, spars = [], [], []
    r_hard, r_soft, p_hard, p_soft = [], [], [], []

    for x, x_cf in zip(X_orig, CFs):
        prox_l1.append(proximity(x, x_cf, norm="l1"))
        prox_l2.append(proximity(x, x_cf, norm="l2"))
        spars.append(sparsity(x, x_cf))

        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mechanisms)
        p = pearl.score(x, x_cf, t, graph, mechanisms)
        r_hard.append(r["hard"])
        r_soft.append(r["soft"])
        p_hard.append(p["hard"])
        p_soft.append(p["soft"])

    ood_scores = np.atleast_1d(ood_plausibility(X_train, CFs))
    sparsity_mean = float(np.mean(spars))

    return {
        "n": int(len(CFs)),
        "validity": validity(CFs, model, target_class),
        "proximity_l1": float(np.mean(prox_l1)),
        "proximity_l2": float(np.mean(prox_l2)),
        "sparsity": sparsity_mean,
        "frac_altered": float(1.0 - sparsity_mean),
        "ood": float(np.mean(ood_scores)),
        "cf_faith_rollout_hard": float(np.mean(r_hard)),
        "cf_faith_rollout_soft": float(np.mean(r_soft)),
        "cf_faith_pearl_hard": float(np.mean(p_hard)),
        "cf_faith_pearl_soft": float(np.mean(p_soft)),
    }


def _generate_batch(method, X, model, graph, mechanisms):
    """Call ``method.generate_batch`` with the right signature.

    CARLA-style recourse needs ``graph``/``mechanisms``; Wachter/DiCE do not.
    We inspect the signature rather than special-casing class names.
    """
    params = inspect.signature(method.generate_batch).parameters
    if "graph" in params or "mechanisms" in params:
        return method.generate_batch(X, model, graph, mechanisms)
    return method.generate_batch(X, model)


def shift_vr(
    model,
    methods: dict,
    X_base_test: np.ndarray,
    X_shift_test: np.ndarray,
    graph: np.ndarray,
    mechanisms: list,
    target_class: int = 1,
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
        (Wachter/DiCE: ``(X, model)``; CARLA: ``(X, model, graph, mechanisms)``).
    X_base_test, X_shift_test:
        Test inputs ``(N, T, k)`` from the base and shifted environments.
    graph, mechanisms:
        The (shared) SCM structure, passed to causal methods.
    target_class:
        Desired output class for validity.

    Returns
    -------
    dict
        ``{name: {"validity_base", "validity_shift", "shift_vr"}}``. ``shift_vr``
        is ``validity_shift / validity_base``, or ``nan`` when no base CF is
        valid (documented denominator-zero sentinel).
    """
    results: dict = {}
    for name, method in methods.items():
        cf_base = _generate_batch(method, X_base_test, model, graph, mechanisms)
        cf_shift = _generate_batch(method, X_shift_test, model, graph, mechanisms)
        v_base = validity(cf_base, model, target_class)
        v_shift = validity(cf_shift, model, target_class)
        ratio = v_shift / v_base if v_base > 0 else float("nan")
        results[name] = {
            "validity_base": v_base,
            "validity_shift": v_shift,
            "shift_vr": ratio,
        }
    return results
