"""
Axis B — Graph Quality Metrics.

Metrics:
  SHD        : Structural Hamming Distance between inferred and ground-truth adjacency
  Lag Acc    : Fraction of edges where the correct lag is identified (when SHD = 0)
  AUC-ROC    : Edge-level AUC over binary edge presence across the graph
  TV-Conf    : Total-Variation confounding score (unused confounders)
  GraphErrDecomp: CF-faith against ground-truth graph vs. inferred graph (for CITRIS/iCITRIS)

Reference:
  Peters et al. (2013), Identifiability of Gaussian SEM; Runge et al. (2019),
  Detecting and quantifying causal associations in large nonlinear time series datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def shd(adj_true: np.ndarray, adj_pred: np.ndarray) -> int:
    """
    Structural Hamming Distance.

    Parameters
    ----------
    adj_true : (M, M[, max_lag]) binary true adjacency
    adj_pred : same shape as adj_true

    Returns
    -------
    int — number of wrong edge decisions (false positives + false negatives)
    """
    return int(np.sum(adj_true.astype(bool) != adj_pred.astype(bool)))


def lag_accuracy(adj_true: np.ndarray, adj_pred: np.ndarray) -> float:
    """
    Fraction of true edges for which the correct lag is predicted.

    Parameters
    ----------
    adj_true : (M, M, max_lag)
    adj_pred : (M, M, max_lag)

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
        # Correct lag: predicted edge at (i, j, lag) is 1
        if adj_pred[i, j, lag] == 1:
            correct += 1
    return float(correct / len(true_edges))


def graph_auc(adj_true: np.ndarray, score_matrix: np.ndarray) -> float:
    """
    AUC-ROC for edge detection.

    Parameters
    ----------
    adj_true      : (M, M) or (M, M, max_lag) binary ground-truth
    score_matrix  : same shape, continuous edge scores (e.g. mutual information)

    Returns
    -------
    float in [0, 1]
    """
    y_true = adj_true.flatten().astype(int)
    y_score = score_matrix.flatten().astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def tv_confounding(X: np.ndarray, adj_true: np.ndarray) -> float:
    """
    Total-Variation Confounding Score.

    For each pair of channels (i, j) with no direct edge in adj_true,
    measure the TV-distance of their marginal distributions. High TV between
    non-adjacent channels signals spurious confounding.

    Parameters
    ----------
    X        : (N, T, M)
    adj_true : (M, M) lag-aggregated binary adjacency

    Returns
    -------
    float — mean TV over non-adjacent pairs; 0.0 if fully connected
    """
    M = adj_true.shape[0]
    tv_scores = []
    ch_means = X.mean(axis=1)  # (N, M) time-average per channel

    for i in range(M):
        for j in range(i + 1, M):
            if adj_true[i, j] == 0 and adj_true[j, i] == 0:
                # TV via histogram overlap
                x_i = ch_means[:, i]
                x_j = ch_means[:, j]
                bins = np.linspace(
                    min(x_i.min(), x_j.min()),
                    max(x_i.max(), x_j.max()) + 1e-6, 31
                )
                h_i, _ = np.histogram(x_i, bins=bins, density=True)
                h_j, _ = np.histogram(x_j, bins=bins, density=True)
                dx = bins[1] - bins[0]
                tv = 0.5 * np.sum(np.abs(h_i - h_j)) * dx
                tv_scores.append(tv)

    return float(np.mean(tv_scores)) if tv_scores else 0.0


def graph_error_decomposition(cf_faith_vs_gt: float,
                               cf_faith_vs_inferred: float) -> dict[str, float]:
    """
    Decompose CF-faith drop into graph-estimation error vs. propagation failure.

    Parameters
    ----------
    cf_faith_vs_gt       : CF-faith when using ground-truth graph to produce X'_CF
    cf_faith_vs_inferred : CF-faith when using inferred graph to produce X'_CF

    Returns
    -------
    dict with:
      'graph_error'      : drop in CF-faith attributable to wrong graph
      'propagation_error': residual after accounting for graph error
    """
    graph_err = cf_faith_vs_gt - cf_faith_vs_inferred
    return {
        "cf_faith_gt": cf_faith_vs_gt,
        "cf_faith_inferred": cf_faith_vs_inferred,
        "graph_error": float(graph_err),
        "propagation_error": float(1.0 - cf_faith_vs_gt),
    }


def compute_axis_b(adj_true_lagged: np.ndarray,
                   adj_pred_lagged: np.ndarray,
                   score_matrix: np.ndarray | None = None,
                   X: np.ndarray | None = None,
                   cf_faith_gt: float | None = None,
                   cf_faith_inferred: float | None = None) -> dict[str, float]:
    """
    Aggregate Axis B metrics.

    Parameters
    ----------
    adj_true_lagged : (M, M, max_lag) ground-truth lagged adjacency
    adj_pred_lagged : (M, M, max_lag) predicted lagged adjacency
    score_matrix    : (M, M, max_lag) optional continuous edge scores for AUC
    X               : (N, T, M) optional for TV-Confounding
    cf_faith_gt     : optional for graph-error decomposition
    cf_faith_inferred : optional for graph-error decomposition
    """
    M = adj_true_lagged.shape[0]
    adj_true_bin = (adj_true_lagged > 0).astype(int)
    adj_pred_bin = (adj_pred_lagged > 0).astype(int)

    results: dict[str, float] = {
        "SHD": float(shd(adj_true_bin, adj_pred_bin)),
        "LagAcc": lag_accuracy(adj_true_lagged, adj_pred_lagged),
    }

    if score_matrix is not None:
        results["AUC"] = graph_auc(adj_true_bin, score_matrix)

    if X is not None:
        adj_agg = adj_true_bin.any(axis=-1).astype(int)  # (M, M)
        results["TV_Confounding"] = tv_confounding(X, adj_agg)

    if cf_faith_gt is not None and cf_faith_inferred is not None:
        decomp = graph_error_decomposition(cf_faith_gt, cf_faith_inferred)
        results.update(decomp)

    return results
