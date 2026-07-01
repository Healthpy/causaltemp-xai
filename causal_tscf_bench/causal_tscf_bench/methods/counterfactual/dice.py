"""
DiCE wrapper — Naive tabular-style baseline.

Treats each (t, j) feature independently (no temporal structure) and optimizes
for validity + diversity. Serves as the null model against which all temporal
methods are compared on CF-faith.

Failure mode (per plan): tabular-style perturbation ignores temporal structure
entirely — no lead-lag propagation, no causal parent-child relationships.

Reference:
  Wachter et al. (2017), Counterfactual Explanations Without Opening the
  Black Box.
"""

from __future__ import annotations

import numpy as np

from ..base import CFExplainer
from ...classifiers.base import TSClassifier


class DiCE(CFExplainer):
    """Tabular DiCE-style gradient-free CF baseline for time series."""

    def __init__(self, n_steps: int = 200, lr: float = 0.05, lambda_prox: float = 1.0):
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_prox = lambda_prox
        self._x_std: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        self._x_std = X_train.std(axis=0) + 1e-6

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        rng = np.random.default_rng()
        x_cf = x.copy().astype(np.float64)
        std = self._x_std if self._x_std is not None else np.ones_like(x)

        best = x_cf.copy()
        best_conf = 0.0

        for step in range(self.n_steps):
            proba = classifier.predict_proba(x_cf[np.newaxis])[0]
            conf = proba[target_class]
            if conf > best_conf:
                best_conf = conf
                best = x_cf.copy()
            if conf > 0.8:
                break
            # Random gradient-free perturbation in direction of increasing confidence
            delta = rng.normal(0, self.lr, x_cf.shape) * std
            x_try = x_cf + delta
            conf_try = classifier.predict_proba(x_try[np.newaxis])[0, target_class]
            # Accept if improves confidence with proximity penalty
            prox_pen = self.lambda_prox * np.abs(x_try - x).mean()
            if conf_try - prox_pen > conf - self.lambda_prox * np.abs(x_cf - x).mean():
                x_cf = x_try

        return best
