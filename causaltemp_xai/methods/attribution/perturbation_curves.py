"""Deletion / insertion curves — a lightweight saliency-quality proxy (WP3).

Ranks ``(t, feature)`` cells by ``|attribution|`` and either progressively
*masks* them to the baseline (deletion) or *reveals* them from the baseline
(insertion), recording the model's confidence on ``target_class`` at each step.

This is intentionally a foil, not a full attribution benchmark. The headline
diagnostic is that a saliency map can score well here (sharp deletion drop /
insertion rise) while the same model's counterfactuals score poorly on CF-faith.
"""

from __future__ import annotations

import numpy as np


def _confidence(model, arr: np.ndarray, target_class: int) -> float:
    """P(target_class) for a single ``(T, k)`` instance."""
    proba = model.predict_proba(arr[np.newaxis])
    return float(proba[0, target_class])


def _curve(
    model,
    x: np.ndarray,
    attribution: np.ndarray,
    target_class: int,
    baseline: np.ndarray | None,
    n_steps: int | None,
    mode: str,
) -> tuple[np.ndarray, float]:
    x_arr = np.asarray(x, dtype=np.float32)
    if baseline is None:
        baseline = np.zeros_like(x_arr)
    base_arr = np.asarray(baseline, dtype=np.float32)

    flat_x = x_arr.ravel()
    flat_b = base_arr.ravel()
    n_cells = flat_x.size
    # Most important cells first (descending |attribution|).
    order = np.argsort(-np.abs(np.asarray(attribution, dtype=float).ravel()))

    if n_steps is None:
        n_steps = n_cells

    probs = np.empty(n_steps + 1, dtype=float)
    for i in range(n_steps + 1):
        k_cells = int(round(i * n_cells / n_steps))
        if mode == "deletion":
            cur = flat_x.copy()
            cur[order[:k_cells]] = flat_b[order[:k_cells]]
        else:  # insertion
            cur = flat_b.copy()
            cur[order[:k_cells]] = flat_x[order[:k_cells]]
        probs[i] = _confidence(model, cur.reshape(x_arr.shape), target_class)

    # Normalised AUC over the fraction axis [0, 1].
    auc = float(np.trapezoid(probs) / n_steps)
    return probs, auc


def deletion_curve(
    model,
    x: np.ndarray,
    attribution: np.ndarray,
    target_class: int,
    baseline: np.ndarray | None = None,
    n_steps: int | None = None,
) -> tuple[np.ndarray, float]:
    """Progressively mask the highest-attribution cells to the baseline.

    A good saliency map produces a steep early confidence drop → **low** AUC.

    Returns ``(curve, auc)`` where ``curve`` has length ``n_steps + 1`` and
    starts at full confidence ``P(target_class | x)``.
    """
    return _curve(model, x, attribution, target_class, baseline, n_steps, "deletion")


def insertion_curve(
    model,
    x: np.ndarray,
    attribution: np.ndarray,
    target_class: int,
    baseline: np.ndarray | None = None,
    n_steps: int | None = None,
) -> tuple[np.ndarray, float]:
    """Progressively reveal the highest-attribution cells from the baseline.

    A good saliency map produces a steep early confidence rise → **high** AUC.

    Returns ``(curve, auc)`` where ``curve`` has length ``n_steps + 1`` and
    ends at full confidence ``P(target_class | x)``.
    """
    return _curve(model, x, attribution, target_class, baseline, n_steps, "insertion")
