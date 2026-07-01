"""
CONFETTI CF wrapper — NUN-search + genetic subsequence replacement.

Wraps cfts_repo's confetti_genetic_cf. Finds the nearest unlike neighbor and
optimizes a binary mask indicating which subsequences to replace from the NUN.

confetti's model_predict auto-handles (T, M) input (transposes when T > M),
so we pass our data directly without extra transposition.

Reference:
  Bahri et al. (2025), CONFETTI: COuNterfactual Explanations For Time
  Series. arXiv:2511.13237.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from ..base import CFExplainer
from ...classifiers.base import TSClassifier

_CFTS = pathlib.Path(__file__).parents[4] / "third_party" / "cfts_repo"
if str(_CFTS) not in sys.path:
    sys.path.insert(0, str(_CFTS))


class ConfettiCF(CFExplainer):
    """Genetic subsequence-replacement CF explainer wrapping cfts confetti."""

    def __init__(
        self,
        theta: float = 0.51,
        max_iterations: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.1,
    ):
        self.theta = theta
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self._X_train: np.ndarray | None = None
        self._Y_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        self._X_train = X_train  # (N, T, M)
        self._Y_train = classifier.predict(X_train)

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        from cfts.cf_confetti.confetti import confetti_genetic_cf

        # confetti auto-transposes 2D input when rows > cols (T > M),
        # so (T, M) and (N, T, M) can be passed directly.
        result = confetti_genetic_cf(
            x,
            classifier.model,
            self._X_train,
            reference_labels=self._Y_train,
            target=target_class,
            theta=self.theta,
            max_iterations=self.max_iterations,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            verbose=False,
        )

        if isinstance(result, tuple):
            cf, _ = result
        else:
            cf = result

        if cf is None:
            return x.copy()
        return np.array(cf, dtype=np.float32)
