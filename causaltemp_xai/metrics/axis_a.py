"""Axis A - Structure (graph) metrics.

Full port from causal_tscf_bench/metrics/axis_a.py.

Metrics:
  SHD        : Structural Hamming Distance between inferred and ground-truth adjacency
  LagAcc     : Fraction of true edges recovered at the correct lag (recall-only)
  LagF1      : F1 over lagged edges (precision-aware companion to LagAcc)
  AUC-ROC    : Edge-level AUC over binary edge presence across the graph
  ResidualDep: |Pearson r| of mechanism residuals between non-adjacent channels
               (replaces the retired TV-confounding score, fix #1 2026-07-18)
  GraphErrDecomp: CF-faith against ground-truth graph vs. inferred graph

Reference:
  Peters et al. (2013), Identifiability of Gaussian SEM; Runge et al. (2019),
  Detecting and quantifying causal associations in large nonlinear time series datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def shd(adj_true: np.ndarray, adj_pred: np.ndarray) -> int:
    """Structural Hamming Distance.

    Parameters
    ----------
    adj_true : (k, k[, max_lag]) binary true adjacency
    adj_pred : same shape as adj_true

    Returns
    -------
    int -- number of wrong edge decisions (false positives + false negatives)
    """
    return int(np.sum(adj_true.astype(bool) != adj_pred.astype(bool)))


def lag_accuracy(adj_true: np.ndarray, adj_pred: np.ndarray) -> float:
    """Fraction of true edges for which the correct lag is predicted.

    Parameters
    ----------
    adj_true : (k, k, max_lag)
    adj_pred : (k, k, max_lag)

    Returns
    -------
    float in [0, 1]; nan if no true edges
    """
    true_edges = np.argwhere(adj_true)
    if len(true_edges) == 0:
        return float("nan")
    correct = 0
    for e in true_edges:
        i, j, lag = e
        if adj_pred[i, j, lag] == 1:
            correct += 1
    return float(correct / len(true_edges))


def graph_auc(adj_true: np.ndarray, score_matrix: np.ndarray) -> float:
    """AUC-ROC for edge detection.

    Parameters
    ----------
    adj_true      : (k, k) or (k, k, max_lag) binary ground-truth
    score_matrix  : same shape, continuous edge scores

    Returns
    -------
    float in [0, 1]
    """
    y_true = adj_true.flatten().astype(int)
    y_score = score_matrix.flatten().astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def residual_dependence(X: np.ndarray, adj_true: np.ndarray, mechanism=None) -> float:
    """Residual-Dependence Score — unexplained association between non-adjacent channels.

    Replaces ``tv_confounding`` (metric-quality fix #1, 2026-07-18).
    Confounding / unmodeled structure manifests as **dependence** between
    channels the graph says are not directly connected — not as dissimilarity
    of their marginal distributions, which is what the retired TV formulation
    measured (two independent channels with different scales scored high TV,
    and a confounder-free SCM read ~0.6).

    When ``mechanism`` is given, dependence is measured on the **abducted
    mechanism residuals** ``eps[t] = x[t] - f(window_t)`` (exact under the
    benchmark's additive-noise SCMs): after the graph explains what it can,
    any remaining correlation between non-adjacent channels' innovations is
    genuine unmodeled association. Residuals at ``t < L`` are excluded (they
    absorb initial conditions — see ``abduct_noise``). Without a mechanism the
    raw channel values are used (weaker: parent-mediated association is then
    not removed and can inflate the score).

    Parameters
    ----------
    X         : (N, T, k)
    adj_true  : (k, k) lag-aggregated binary adjacency
    mechanism : optional Mechanism — enables residual (recommended) mode

    Returns
    -------
    float -- mean |Pearson r| over non-adjacent channel pairs, in [0, 1];
    ~0 for a well-specified confounder-free SCM. 0.0 if fully connected.
    """
    X = np.asarray(X, dtype=float)
    k = adj_true.shape[0]

    if mechanism is not None:
        from causaltemp_xai.benchmarks.structural_cf import abduct_noise

        L = mechanism.L
        R = np.stack([abduct_noise(x, mechanism)[L:] for x in X])  # (N, T-L, k)
    else:
        R = X
    flat = R.reshape(-1, k)  # pool instances and time

    scores = []
    for i in range(k):
        for j in range(i + 1, k):
            if adj_true[i, j] == 0 and adj_true[j, i] == 0:
                a, b = flat[:, i], flat[:, j]
                sa, sb = a.std(), b.std()
                if sa < 1e-12 or sb < 1e-12:
                    scores.append(0.0)
                    continue
                r = float(np.corrcoef(a, b)[0, 1])
                scores.append(abs(r) if np.isfinite(r) else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def lagged_edge_f1(adj_true: np.ndarray, adj_pred: np.ndarray) -> float:
    """F1 over lagged edges — the precision-aware companion to ``lag_accuracy``.

    ``lag_accuracy`` is recall-only: a predicted **complete** graph (every
    edge at every lag) scores 1.0 (metric-quality fix #7, 2026-07-18). F1
    keeps the lag-resolved recall but charges for spurious edges.

    Parameters
    ----------
    adj_true : (k, k, max_lag) binary
    adj_pred : (k, k, max_lag) binary

    Returns
    -------
    float in [0, 1]; nan if the true graph has no edges.
    """
    t = np.asarray(adj_true).astype(bool)
    p = np.asarray(adj_pred).astype(bool)
    n_true = int(t.sum())
    if n_true == 0:
        return float("nan")
    tp = int((t & p).sum())
    n_pred = int(p.sum())
    if n_pred == 0 or tp == 0:
        return 0.0
    precision = tp / n_pred
    recall = tp / n_true
    return float(2 * precision * recall / (precision + recall))


def graph_error_decomposition(cf_faith_vs_gt: float, cf_faith_vs_inferred: float) -> dict:
    """Decompose CF-faith drop into graph-estimation error vs. propagation failure.

    Sign convention (metric-quality fix #10, 2026-07-18): this is bookkeeping,
    not a decomposition into non-negative parts. ``graph_error =
    cf_faith_gt - cf_faith_inferred`` **can be negative** — an inferred graph
    can outscore the ground truth by chance on a finite CF sample. Report the
    signed value; do not clip.

    Parameters
    ----------
    cf_faith_vs_gt       : CF-faith when using ground-truth graph
    cf_faith_vs_inferred : CF-faith when using inferred graph

    Returns
    -------
    dict with: cf_faith_gt, cf_faith_inferred, graph_error (signed),
    propagation_error
    """
    graph_err = cf_faith_vs_gt - cf_faith_vs_inferred
    return {
        "cf_faith_gt": cf_faith_vs_gt,
        "cf_faith_inferred": cf_faith_vs_inferred,
        "graph_error": float(graph_err),
        "propagation_error": float(1.0 - cf_faith_vs_gt),
    }


def compute_axis_a(
    adj_true_lagged: np.ndarray,
    adj_pred_lagged: np.ndarray,
    score_matrix: np.ndarray | None = None,
    X: np.ndarray | None = None,
    cf_faith_gt: float | None = None,
    cf_faith_inferred: float | None = None,
    mechanism=None,
) -> dict:
    """Aggregate Axis A metrics.

    Parameters
    ----------
    adj_true_lagged : (k, k, max_lag) ground-truth lagged adjacency
    adj_pred_lagged : (k, k, max_lag) predicted lagged adjacency
    score_matrix    : (k, k, max_lag) optional continuous edge scores for AUC
    X               : (N, T, k) optional for the residual-dependence diagnostic
    cf_faith_gt     : optional for graph-error decomposition
    cf_faith_inferred : optional for graph-error decomposition
    mechanism       : optional Mechanism — residual (recommended) mode for
                      the dependence diagnostic

    Returns
    -------
    dict with keys: SHD, LagAcc, LagF1, (AUC), (ResidualDep), (cf_faith_gt, ...)
    """
    adj_true_bin = (adj_true_lagged > 0).astype(int)
    adj_pred_bin = (adj_pred_lagged > 0).astype(int)

    results: dict = {
        "SHD": float(shd(adj_true_bin, adj_pred_bin)),
        "LagAcc": lag_accuracy(adj_true_lagged, adj_pred_lagged),
        "LagF1": lagged_edge_f1(adj_true_bin, adj_pred_bin),
    }

    if score_matrix is not None:
        results["AUC"] = graph_auc(adj_true_bin, score_matrix)

    if X is not None:
        adj_agg = adj_true_bin.any(axis=-1).astype(int)  # (k, k)
        results["ResidualDep"] = residual_dependence(X, adj_agg, mechanism=mechanism)

    if cf_faith_gt is not None and cf_faith_inferred is not None:
        decomp = graph_error_decomposition(cf_faith_gt, cf_faith_inferred)
        results.update(decomp)

    return results
