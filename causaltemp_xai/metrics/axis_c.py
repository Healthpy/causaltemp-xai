"""Axis-C counterfactual evaluation metrics.

Implements the four standard axes for evaluating counterfactual quality:

* **Validity**        – CF achieves the desired class under the black-box model.
* **Proximity**       – CF is close to the original instance (L1 or L2).
* **Sparsity**        – few features differ between the original and the CF.
* **OOD Plausibility** – CF lies within the training distribution, estimated
                          via sklearn's IsolationForest.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.ensemble import IsolationForest

from causaltemp_xai.scm.intervention import INTERVENTION_TOL


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def validity(
    x_cf: np.ndarray,
    model,
    target_class: int,
) -> float:
    """Fraction of counterfactuals predicted as ``target_class``.

    This is the standard CF *validity* (flip-rate): the share of proposed
    counterfactuals that the black-box model actually assigns to the desired
    class.  A value of ``1.0`` means every CF achieved the target.

    Parameters
    ----------
    x_cf:
        Counterfactual instance(s).  Either a single instance of shape
        ``(T, k)`` or a batch of shape ``(N, T, k)``.  A batch dimension is
        added automatically for a single ``(T, k)`` instance.
    model:
        Classifier.  Either an object exposing ``predict`` (e.g.
        :class:`~causaltemp_xai.classifiers.LSTMClassifier`) or a plain callable
        mapping a ``(N, T, k)`` batch to integer labels of shape ``(N,)``.
    target_class:
        The desired output class the counterfactuals should achieve.

    Returns
    -------
    float
        Validity (flip-rate) in ``[0, 1]``.
    """
    predict = getattr(model, "predict", model)
    batch = np.asarray(x_cf, dtype=float)
    if batch.ndim == 2:  # single (T, k) → add batch dim
        batch = batch[np.newaxis]
    preds = np.asarray(predict(batch)).reshape(-1)
    return float(np.mean(preds == target_class))


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------


def proximity(
    x_original: np.ndarray,
    x_cf: np.ndarray,
    norm: Literal["l1", "l2"] = "l1",
) -> float:
    """Distance between the original instance and the counterfactual.

    Parameters
    ----------
    x_original:
        Original time series, shape ``(T, k)`` or ``(k,)``.
    x_cf:
        Counterfactual, same shape as ``x_original``.
    norm:
        ``"l1"`` (Manhattan) or ``"l2"`` (Euclidean).

    Returns
    -------
    float
        Scalar distance.  Lower is closer (better).
    """
    x_orig = np.asarray(x_original, dtype=float).ravel()
    x_cf_arr = np.asarray(x_cf, dtype=float).ravel()
    diff = x_orig - x_cf_arr
    if norm == "l1":
        return float(np.sum(np.abs(diff)))
    elif norm == "l2":
        return float(np.sqrt(np.sum(diff ** 2)))
    else:
        raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------


def sparsity(
    x_original: np.ndarray,
    x_cf: np.ndarray,
    tol: float = 1e-6,
) -> float:
    """Fraction of features that are unchanged between original and CF.

    A higher sparsity score (closer to 1) means fewer features were modified,
    which is generally desirable for interpretability.

    Parameters
    ----------
    x_original:
        Original instance, shape ``(T, k)`` or ``(k,)``.
    x_cf:
        Counterfactual instance, same shape.
    tol:
        Absolute tolerance below which a difference counts as zero.

    Returns
    -------
    float
        Score in ``[0, 1]``.  1 means no features were changed.
    """
    x_orig = np.asarray(x_original, dtype=float).ravel()
    x_cf_arr = np.asarray(x_cf, dtype=float).ravel()
    n_unchanged = int(np.sum(np.abs(x_orig - x_cf_arr) <= tol))
    return n_unchanged / len(x_orig)


# ---------------------------------------------------------------------------
# OOD Plausibility
# ---------------------------------------------------------------------------


def ood_plausibility(
    x_train: np.ndarray,
    x_cf: np.ndarray,
    method: Literal["if"] = "if",
    contamination: float = 0.05,
    random_state: int = 0,
) -> float:
    """Estimate whether the counterfactual lies within the training distribution.

    Uses an IsolationForest trained on ``x_train`` to score ``x_cf``.
    The IsolationForest ``decision_function`` returns higher values for
    in-distribution points; we return that score directly so callers can
    interpret it (positive = plausible, negative = anomalous).

    Parameters
    ----------
    x_train:
        Training instances, shape ``(N, T, k)`` or ``(N, k)``.  Used to fit
        the IsolationForest.
    x_cf:
        Counterfactual instance to evaluate, shape ``(T, k)`` or ``(k,)``.
        A batch of CFs of shape ``(M, T, k)`` is also accepted.
    method:
        Currently only ``"if"`` (IsolationForest) is supported.
    contamination:
        Expected fraction of outliers in the training set.  Passed directly
        to :class:`sklearn.ensemble.IsolationForest`.
    random_state:
        Random seed for IsolationForest reproducibility.

    Returns
    -------
    float or ndarray
        IsolationForest anomaly score(s).  Higher = more plausible (in-dist).
        Returns a scalar for a single CF, or an array for a batch.
    """
    if method != "if":
        raise ValueError(f"method must be 'if', got {method!r}")

    X_tr = np.asarray(x_train, dtype=float)
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X_tr_flat)

    x_cf_arr = np.asarray(x_cf, dtype=float)
    if x_cf_arr.ndim == X_tr.ndim - 1:
        # Single instance
        x_cf_flat = x_cf_arr.ravel().reshape(1, -1)
        scores = clf.decision_function(x_cf_flat)
        return float(scores[0])
    else:
        # Batch
        x_cf_flat = x_cf_arr.reshape(x_cf_arr.shape[0], -1)
        return clf.decision_function(x_cf_flat)


# ---------------------------------------------------------------------------
# Bench-ported additions (Axis C expansion)
# ---------------------------------------------------------------------------


def proximity_dtw(
    X: np.ndarray,
    X_cf: np.ndarray,
    normalize: bool = True,
) -> float:
    """Mean DTW distance between factual and CF, optionally normalized by std.

    Ported from causal_tscf_bench/metrics/axis_c.py.

    Parameters
    ----------
    X    : (N, T, k) factual instances (batch) or (T, k) single instance
    X_cf : (N, T, k) CF instances (batch) or (T, k) single instance
    normalize: if True, divide by std(X)

    Returns
    -------
    float -- mean DTW distance
    """
    try:
        from tslearn.metrics import dtw as _dtw
    except ImportError as exc:
        raise ImportError("proximity_dtw requires tslearn: pip install tslearn") from exc

    X_arr = np.asarray(X, dtype=float)
    X_cf_arr = np.asarray(X_cf, dtype=float)
    if X_arr.ndim == 2:
        X_arr = X_arr[np.newaxis]
        X_cf_arr = X_cf_arr[np.newaxis]
    N = X_arr.shape[0]
    dists = [_dtw(X_arr[i], X_cf_arr[i]) for i in range(N)]
    raw = float(np.mean(dists))
    if normalize:
        return raw / (float(X_arr.std()) + 1e-8)
    return raw


def trsi(X_cf: np.ndarray, X: np.ndarray) -> float:
    """Temporal Relevance Smoothness Index (TRSI).

    Measures mean L2 norm of the second-order perturbation difference over time.
    Low TRSI = smooth temporal changes. High TRSI = abrupt discontinuities.

    TRSI = (1/(T-1)) sum_t ||Delta_{t+1} - Delta_t||_2  where Delta_t = X_cf_t - X_t.

    The L2 norm is over channels k; then averaged over T-1 steps and N instances.

    Parameters
    ----------
    X_cf : (N, T, k) or (T, k) -- counterfactuals
    X    : (N, T, k) or (T, k) -- factuals

    Returns
    -------
    float
    """
    X_cf_arr = np.asarray(X_cf, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if X_cf_arr.ndim == 2:
        X_cf_arr = X_cf_arr[np.newaxis]
        X_arr = X_arr[np.newaxis]
    delta = X_cf_arr - X_arr                             # (N, T, k)
    d_delta = np.diff(delta, axis=1)                     # (N, T-1, k) -- first difference
    l2_per_step = np.sqrt((d_delta ** 2).sum(axis=-1))   # (N, T-1)
    return float(l2_per_step.mean())


def ivr(X_cf: np.ndarray, X: np.ndarray, T_int: int, eps: float = INTERVENTION_TOL) -> float:
    """Irreversibility-Violation-Rate (IVR).

    Fraction of CF instances that modify any time step t < T_int.
    Under causal faithfulness, only t >= T_int should change.

    The per-element threshold ``eps`` defaults to
    :data:`~causaltemp_xai.scm.intervention.INTERVENTION_TOL` — the same
    predicate ``derive_intervention_t`` uses to define the intervention
    timestep (M1 fix, 2026-07-07; previously ``1e-4``, which flagged
    float-scale reconstruction noise that the t0 definition itself certified
    as "unchanged" — the CELS IVR=1.0 artifact).

    NOTE: when ``T_int`` is *derived* via ``derive_intervention_t`` with the
    same tolerance (as the phased pipeline does), IVR is 0 by construction —
    in that wiring it is a pipeline-consistency canary, not a discriminative
    score. It is discriminative only for methods that *declare* their own
    intervention time (e.g. oracle structural CFs).

    Parameters
    ----------
    X_cf  : (N, T, k) or (T, k)
    X     : (N, T, k) or (T, k)
    T_int : intervention time step (changes before this are violations)
    eps   : per-element threshold below which changes are considered zero

    Returns
    -------
    float in [0, 1]
    """
    X_cf_arr = np.asarray(X_cf, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if X_cf_arr.ndim == 2:
        X_cf_arr = X_cf_arr[np.newaxis]
        X_arr = X_arr[np.newaxis]
    pre_int_change = np.abs(X_cf_arr[:, :T_int, :] - X_arr[:, :T_int, :]) > eps
    violation_mask = pre_int_change.any(axis=(1, 2))   # (N,)
    return float(violation_mask.mean())


def ood_mahalanobis(X_cf: np.ndarray, X_train: np.ndarray) -> float:
    """Mahalanobis distance OOD score (reference implementation).

    NOTE: O(D^3) -- breaks for high-dimensional inputs. Use ood_plausibility()
    for production use. This function is provided for reference/comparison.

    Parameters
    ----------
    X_cf    : (N, T, k) counterfactuals to score
    X_train : (N_train, T, k) training distribution

    Returns
    -------
    float -- mean Mahalanobis distance
    """
    N, T, k = X_cf.shape
    X_cf_flat = X_cf.reshape(N, -1).astype(np.float64)
    X_tr_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64)

    mu = X_tr_flat.mean(axis=0)
    diff = X_tr_flat - mu
    cov = diff.T @ diff / max(len(X_tr_flat) - 1, 1) + 1e-6 * np.eye(T * k)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    dists = []
    for x in X_cf_flat:
        d = x - mu
        dists.append(float(np.sqrt(d @ cov_inv @ d)))
    return float(np.mean(dists))


def compute_axis_c(X: np.ndarray, X_cf_exp: np.ndarray,
                   X_train: np.ndarray, classifier,
                   target_class: int,
                   T_int: int | None = None) -> dict:
    """Compute all Axis C metrics.

    Parameters
    ----------
    X          : (N, T, k) factual instances
    X_cf_exp   : (N, T, k) explainer CFs
    X_train    : (N_train, T, k) training distribution
    classifier : classifier with .predict()
    target_class : int
    T_int      : intervention time step for IVR (skipped if None)

    Returns
    -------
    dict with Validity, Proximity_L1, Proximity_L2, Sparsity, OOD, TRSI,
    and IVR (if T_int provided).
    """
    results: dict = {}
    results["Validity"] = validity(X_cf_exp, classifier, target_class)
    results["Proximity_L1"] = float(np.mean([
        proximity(X[i], X_cf_exp[i], norm="l1") for i in range(len(X))
    ]))
    results["Proximity_L2"] = float(np.mean([
        proximity(X[i], X_cf_exp[i], norm="l2") for i in range(len(X))
    ]))
    results["Sparsity"] = float(np.mean([
        sparsity(X[i], X_cf_exp[i]) for i in range(len(X))
    ]))
    results["TRSI"] = trsi(X_cf_exp, X)

    ood_scores = np.atleast_1d(ood_plausibility(X_train, X_cf_exp))
    results["OOD"] = float(np.mean(ood_scores))

    if T_int is not None:
        results["IVR"] = ivr(X_cf_exp, X, T_int)

    return results
