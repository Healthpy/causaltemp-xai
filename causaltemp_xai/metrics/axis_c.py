"""Axis-C counterfactual evaluation metrics.

Implements the four standard axes for evaluating counterfactual quality:

* **Validity**        – CF achieves the desired class under the black-box model.
* **Proximity**       – CF is close to the original instance (L1 or L2).
* **Sparsity**        – few features differ between the original and the CF.
* **OOD Plausibility** – CF lies within the training distribution, estimated
                          via sklearn's IsolationForest.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def validity(x_cf: np.ndarray, model: Callable[[np.ndarray], np.ndarray]) -> float:
    """Check whether the counterfactual achieves a *different* class than it
    would if it were the original (i.e. the model's prediction flipped).

    In the common single-instance usage the caller compares ``model(x_cf)``
    against the original prediction.  This function wraps that call so all
    Axis-C metrics share the same interface.

    Parameters
    ----------
    x_cf:
        Counterfactual instance(s).  Either a single instance of shape
        ``(T, k)`` / ``(k,)`` or a batch of shape ``(N, T, k)`` / ``(N, k)``.
        A batch dimension is added automatically for single instances.
    model:
        Black-box classifier callable.  Must accept a batch array of shape
        ``(N, ...)`` and return predicted class labels of shape ``(N,)``.

    Returns
    -------
    float
        Mean validity across the batch (fraction of CFs predicted as the
        model's output class, which the caller can compare against a target).
    """
    batch = np.asarray(x_cf, dtype=float)
    if batch.ndim == x_cf.ndim - 1 or batch.ndim < 2:
        batch = batch[np.newaxis]  # add batch dim
    preds = np.asarray(model(batch))
    # Return raw predictions so callers can compare against their target
    return preds


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
