"""
Glacier CF wrapper — deep latent space counterfactual.

Wraps cfts_repo's glacier_cf. Optimizes in the learned latent space of
an autoencoder to flip the classifier while enforcing plausibility.

Reference:
  Zhao & Doshi-Velez (2021), Glacier: Guided Locally Constrained
  Counterfactual Explanations for Time Series.
"""

from __future__ import annotations

import pathlib
import sys
from typing import List, Tuple

import numpy as np

from ..base import CFExplainer
from ...classifiers.base import TSClassifier

_CFTS = pathlib.Path(__file__).parents[4] / "third_party" / "cfts_repo"
if str(_CFTS) not in sys.path:
    sys.path.insert(0, str(_CFTS))


class GlacierCF(CFExplainer):
    """Latent-space CF explainer wrapping cfts glacier_cf."""

    def __init__(
        self,
        lambda_sparse: float = 0.1,
        lambda_proximity: float = 1.0,
        max_iterations: int = 500,
    ):
        self.lambda_sparse = lambda_sparse
        self.lambda_proximity = lambda_proximity
        self.max_iterations = max_iterations
        self._dataset: List[Tuple] = []

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        Y_train = classifier.predict(X_train)
        self._dataset = list(zip(X_train, Y_train.tolist()))

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        from cfts.cf_glacier.glacier import glacier_cf

        result = glacier_cf(
            x,
            self._dataset,
            classifier.model,
            target_class=target_class,
            lambda_sparse=self.lambda_sparse,
            lambda_proximity=self.lambda_proximity,
            max_iterations=self.max_iterations,
        )
        if isinstance(result, tuple):
            cf, _ = result
        else:
            cf = result

        if cf is None:
            return x.copy()
        return np.array(cf, dtype=np.float32)
