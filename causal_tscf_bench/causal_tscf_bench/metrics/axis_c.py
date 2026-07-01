"""
Axis C — Counterfactual Quality Metrics.

Metrics:
  Validity        : fraction of CFs that flip the classifier prediction
  Proximity       : mean DTW distance between X and X'_exp, normalized by std
  Sparsity        : fraction of time steps with zero change (L0 / T)
  OOD_Plausibility: mean anomaly score from Isolation Forest + LOF ensemble
  TRSI            : Temporal Relevance Smoothness Index — (1/(T-1)) Σ_t ||Δ_{t+1}-Δ_t||₂
  CF-faith        : 1 - DTW(X'_exp, X'_CF) / DTW(X, X'_CF)  [normalized DTW]
  IVR             : Irreversibility-Violation-Rate — fraction of CFs modifying t < T_int

Reference:
  Delaney et al. (2021), Counterfactual Explanations for Multivariate Time Series.
  TRSI: proposed in this benchmark (cf. plan Section 4.3); L2 norm over channels per step.
  CF-faith: plan Eq. (3).
  OOD: Isolation Forest (Liu et al., 2008) + LOF (Breunig et al., 2000).
"""

from __future__ import annotations

import numpy as np
from tslearn.metrics import dtw


def validity(X_cf: np.ndarray, classifier, target_class: int) -> float:
    """
    Fraction of CFs predicted as target_class.

    Parameters
    ----------
    X_cf         : (N, T, M)
    classifier   : TSClassifier with .predict()
    target_class : int
    """
    preds = classifier.predict(X_cf)
    return float(np.mean(preds == target_class))


def proximity(X: np.ndarray, X_cf: np.ndarray, normalize: bool = True) -> float:
    """
    Mean DTW distance between factual and CF, optionally normalized by std.

    Parameters
    ----------
    X    : (N, T, M) factual
    X_cf : (N, T, M) CF
    """
    N = X.shape[0]
    dists = []
    for i in range(N):
        d = dtw(X[i], X_cf[i])
        dists.append(d)
    raw = float(np.mean(dists))
    if normalize:
        std = float(X.std()) + 1e-8
        return raw / std
    return raw


def sparsity(X: np.ndarray, X_cf: np.ndarray, eps: float = 1e-4) -> float:
    """
    Fraction of unchanged time steps (L0 sparsity over T × M cells).

    Parameters
    ----------
    X    : (N, T, M)
    X_cf : (N, T, M)
    eps  : threshold below which a change is considered zero
    """
    changed = np.abs(X_cf - X) > eps   # (N, T, M)
    return float(1.0 - changed.mean())


def ood_plausibility(X_cf: np.ndarray, X_train: np.ndarray) -> float:
    """
    OOD plausibility score: mean anomaly score from an ensemble of
    Isolation Forest and Local Outlier Factor fitted on X_train.

    Higher score = more out-of-distribution (less plausible).

    Parameters
    ----------
    X_cf    : (N, T, M)
    X_train : (N_train, T, M)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    N = X_cf.shape[0]
    X_cf_flat = X_cf.reshape(N, -1).astype(np.float64)
    X_tr_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64)

    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=0)
    iso.fit(X_tr_flat)
    # decision_function: negative for outliers; negate so higher = more OOD
    scores_if = -iso.decision_function(X_cf_flat)

    n_neighbors = min(20, len(X_tr_flat) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_tr_flat)
    scores_lof = -lof.decision_function(X_cf_flat)

    return float((np.mean(scores_if) + np.mean(scores_lof)) / 2)


def ood_mahalanobis(X_cf: np.ndarray, X_train: np.ndarray) -> float:
    """
    Mahalanobis distance fallback (kept for reference; use ood_plausibility).
    O(D³) — breaks for high-dimensional inputs.
    """
    N, T, M = X_cf.shape
    X_cf_flat = X_cf.reshape(N, -1).astype(np.float64)
    X_tr_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64)

    mu = X_tr_flat.mean(axis=0)
    diff = X_tr_flat - mu
    cov = diff.T @ diff / max(len(X_tr_flat) - 1, 1) + 1e-6 * np.eye(T * M)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    dists = []
    for x in X_cf_flat:
        d = x - mu
        dists.append(float(np.sqrt(d @ cov_inv @ d)))
    return float(np.mean(dists))


def trsi(X_cf: np.ndarray, X: np.ndarray) -> float:
    """
    Temporal Relevance Smoothness Index.

    Measures mean L2 norm of the second-order perturbation difference over time.
    Low TRSI = smooth temporal changes (temporally coherent).
    High TRSI = abrupt discontinuities (temporally incoherent).

    TRSI = (1/(T-1)) Σ_t ||Δ_{t+1} - Δ_t||₂   where Δ_t = X'_cf_t - X_t.

    The L2 norm is over channels M; then averaged over T-1 steps and N instances.

    Parameters
    ----------
    X_cf : (N, T, M)
    X    : (N, T, M)
    """
    delta = X_cf - X                               # (N, T, M)
    d_delta = np.diff(delta, axis=1)               # (N, T-1, M) — first difference
    l2_per_step = np.sqrt((d_delta ** 2).sum(axis=-1))   # (N, T-1)
    return float(l2_per_step.mean())


def cf_faith(X_exp: np.ndarray, X_gt_cf: np.ndarray, X: np.ndarray,
             eps: float = 1e-8) -> float:
    """
    CF-faith (normalized DTW) as defined in plan Eq. (3):

      CF-faith = 1 - DTW(X'_exp, X'_CF) / DTW(X, X'_CF)

    Parameters
    ----------
    X_exp    : (T, M) explainer's CF output
    X_gt_cf  : (T, M) analytical ground-truth CF (from causal ladder)
    X        : (T, M) factual instance

    Returns
    -------
    float in (-inf, 1]; higher is better; 1 = perfect causal faithfulness
    """
    d_exp_gt = dtw(X_exp, X_gt_cf)
    d_x_gt = dtw(X, X_gt_cf) + eps
    return float(1.0 - d_exp_gt / d_x_gt)


def cf_faith_batch(X_exp: np.ndarray, X_gt_cf: np.ndarray,
                   X: np.ndarray, percentile: float = 10.0) -> dict[str, float]:
    """
    Batch CF-faith with hard-threshold variant.

    Parameters
    ----------
    X_exp    : (N, T, M)
    X_gt_cf  : (N, T, M)
    X        : (N, T, M)
    percentile: percentile of DTW(X, X'_CF) used as hard threshold ε

    Returns
    -------
    dict: 'CF_faith_mean', 'CF_faith_std', 'CF_faith_hard' (fraction above ε)
    """
    N = X.shape[0]
    scores, d_x_gt_list = [], []

    for i in range(N):
        score = cf_faith(X_exp[i], X_gt_cf[i], X[i])
        scores.append(score)
        d_x_gt_list.append(dtw(X[i], X_gt_cf[i]))

    epsilon = np.percentile(d_x_gt_list, percentile)
    hard_pass = np.mean([dtw(X_exp[i], X_gt_cf[i]) < epsilon for i in range(N)])

    return {
        "CF_faith_mean": float(np.mean(scores)),
        "CF_faith_std": float(np.std(scores)),
        "CF_faith_hard": float(hard_pass),
        "epsilon": float(epsilon),
    }


def ivr(X_cf: np.ndarray, X: np.ndarray, T_int: int, eps: float = 1e-4) -> float:
    """
    Irreversibility-Violation-Rate.

    Fraction of CF instances that modify any time step t < T_int.
    Under causal faithfulness, only t >= T_int should change.

    Parameters
    ----------
    X_cf  : (N, T, M)
    X     : (N, T, M)
    T_int : intervention time step (changes before this are violations)
    eps   : threshold below which changes are considered zero
    """
    pre_int_change = np.abs(X_cf[:, :T_int, :] - X[:, :T_int, :]) > eps
    violation_mask = pre_int_change.any(axis=(1, 2))   # (N,)
    return float(violation_mask.mean())


def compute_axis_c(X: np.ndarray, X_cf_exp: np.ndarray, X_cf_gt: np.ndarray,
                   X_train: np.ndarray, classifier,
                   target_class: int, T_int: int) -> dict[str, float]:
    """
    Compute all Axis C metrics.

    Parameters
    ----------
    X          : (N, T, M) factual
    X_cf_exp   : (N, T, M) explainer's CFs
    X_cf_gt    : (N, T, M) analytical ground-truth CFs
    X_train    : (N_train, T, M) training distribution
    classifier : TSClassifier
    target_class : int
    T_int      : intervention time step
    """
    results: dict[str, float] = {}

    results["Validity"] = validity(X_cf_exp, classifier, target_class)
    results["Proximity"] = proximity(X, X_cf_exp, normalize=True)
    results["Sparsity"] = sparsity(X, X_cf_exp)
    results["OOD"] = ood_plausibility(X_cf_exp, X_train)
    results["TRSI"] = trsi(X_cf_exp, X)
    results["IVR"] = ivr(X_cf_exp, X, T_int)

    faith_results = cf_faith_batch(X_cf_exp, X_cf_gt, X)
    results.update(faith_results)

    return results
