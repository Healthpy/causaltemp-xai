"""
CoMTE CF wrapper — instance-based subsequence substitution.

Wraps cfts_repo's comte_cf. Finds the nearest unlike neighbor and substitutes
subsequences guided by model saliency.

Reference:
  Delaney et al. (2021), Instance-Based Counterfactual Explanations for
  Time Series Classification.
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


class ComteCF(CFExplainer):
    """Instance-substitution CF explainer wrapping cfts comte_cf."""

    def __init__(
        self,
        lambda_reg: float = 0.01,
        lambda_sparse: float = 0.001,
        learning_rate: float = 0.1,
        max_iterations: int = 1000,
    ):
        self.lambda_reg = lambda_reg
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self._dataset: List[Tuple] = []

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        Y_train = classifier.predict(X_train)
        self._dataset = list(zip(X_train, Y_train.tolist()))

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        from cfts.cf_comte.comte import comte_cf

        result = comte_cf(
            x,
            self._dataset,
            classifier.model,
            target_class=target_class,
            lambda_reg=self.lambda_reg,
            lambda_sparse=self.lambda_sparse,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        if isinstance(result, tuple):
            cf, _ = result
        else:
            cf = result

        if cf is None:
            return x.copy()
        return np.array(cf, dtype=np.float32)
