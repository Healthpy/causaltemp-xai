"""
TimeSHAP wrapper — Attribution paradigm (Axes A and D only; no CF output).

Computes SHAP values for time series using a surrogate model that treats each
(t, j) pair as a feature coalition member.

Reference: Bento et al. (2021), TimeSHAP: Explaining Recurrent Models through
Sequence Perturbations.
"""

from __future__ import annotations

import numpy as np

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier


class TimeSHAP(AttributionMethod):
    """Approximate TimeSHAP via random feature masking (Monte Carlo Shapley)."""

    def __init__(self, n_samples: int = 200, baseline: str = "zero"):
        self.n_samples = n_samples
        self.baseline = baseline   # "zero" or "mean"
        self._baseline_val: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> None:
        if self.baseline == "mean":
            self._baseline_val = X_train.mean(axis=0)  # (T, M)

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        T, M = x.shape
        if target_class is None:
            target_class = int(classifier.predict(x[np.newaxis])[0])

        baseline = (self._baseline_val if self._baseline_val is not None
                    else np.zeros_like(x))
        rng = np.random.default_rng()
        phi = np.zeros((T, M), dtype=np.float64)

        for _ in range(self.n_samples):
            # Random coalition mask
            mask = rng.integers(0, 2, size=(T, M)).astype(bool)
            x_masked = np.where(mask, x, baseline)
            x_with = np.where(mask | True, x, baseline)  # add feature
            mask_without = mask.copy()

            # Random feature to attribute
            t_feat = rng.integers(T)
            m_feat = rng.integers(M)
            mask_with = mask.copy()
            mask_with[t_feat, m_feat] = True
            mask_without[t_feat, m_feat] = False

            x_with_feat = np.where(mask_with, x, baseline)
            x_without_feat = np.where(mask_without, x, baseline)

            p_with = classifier.predict_proba(x_with_feat[np.newaxis])[0, target_class]
            p_without = classifier.predict_proba(x_without_feat[np.newaxis])[0, target_class]
            phi[t_feat, m_feat] += (p_with - p_without)

        phi /= self.n_samples
        return phi
