"""
TimeSHAP wrapper — Attribution paradigm (Axes A and D only; no CF output).

Computes SHAP values for time series using a surrogate model that treats each
(t, j) pair as a feature coalition member.

Reference: Bento et al. (2021), TimeSHAP: Explaining Recurrent Models through
Sequence Perturbations.

Ported from causal_tscf_bench/methods/attribution/timeshap.py.
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
            self._baseline_val = X_train.mean(axis=0)  # (T, k)

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        T, k = x.shape
        if target_class is None:
            target_class = int(classifier.predict(x[np.newaxis])[0])

        baseline = (self._baseline_val if self._baseline_val is not None
                    else np.zeros_like(x))
        rng = np.random.default_rng()
        phi = np.zeros((T, k), dtype=np.float64)

        for _ in range(self.n_samples):
            t_feat = rng.integers(T)
            m_feat = rng.integers(k)

            mask_with = rng.integers(0, 2, size=(T, k)).astype(bool)
            mask_with[t_feat, m_feat] = True
            mask_without = mask_with.copy()
            mask_without[t_feat, m_feat] = False

            x_with = np.where(mask_with, x, baseline)
            x_without = np.where(mask_without, x, baseline)

            p_with = classifier.predict_proba(x_with[np.newaxis])[0, target_class]
            p_without = classifier.predict_proba(x_without[np.newaxis])[0, target_class]
            phi[t_feat, m_feat] += (p_with - p_without)

        phi /= self.n_samples
        return phi
