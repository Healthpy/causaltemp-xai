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
