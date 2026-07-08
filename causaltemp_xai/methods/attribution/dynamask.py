"""
Dynamask wrapper — gradient-based temporal mask learning.

Reference: Crabbe & van der Schaar (2021), Explaining Time Series Predictions
with Dynamic Masks. ICML 2021.

Ported from causal_tscf_bench/methods/attribution/dynamask.py.
Note: uses finite-difference saliency as a proxy for the full Dynamask
optimization (which requires the original library).
"""

from __future__ import annotations

import numpy as np

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier


class Dynamask(AttributionMethod):
    """
    Dynamask: learns a soft temporal mask that minimally perturbs
    the input while preserving the classifier decision.

    Uses finite-difference saliency as a proxy when the full
    Dynamask optimization is not available.
    """

    def __init__(self, n_steps: int = 100, lr: float = 0.05,
                 lambda_sparsity: float = 0.1):
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_sparsity = lambda_sparsity

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        T, k = x.shape
        if target_class is None:
            target_class = int(classifier.predict(x[np.newaxis])[0])

        eps = 1e-3
        phi = np.zeros((T, k), dtype=np.float64)
        base_conf = classifier.predict_proba(x[np.newaxis])[0, target_class]

        for t in range(T):
            for m in range(k):
                x_pert = x.copy()
                x_pert[t, m] += eps
                conf_pert = classifier.predict_proba(x_pert[np.newaxis])[0, target_class]
                phi[t, m] = (conf_pert - base_conf) / eps

        return phi
