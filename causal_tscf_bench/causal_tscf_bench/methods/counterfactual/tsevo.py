"""
TSEvo CF wrapper — evolutionary time series counterfactual.

Wraps cfts_repo's tsevo_cf. Evolves counterfactuals via genetic algorithm
combining proximity, sparsity, and validity objectives.

Reference:
  Hollig et al. (2022), TSEvo: Evolutionary Counterfactual Explanations for
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


class TSEvoCF(CFExplainer):
    """Evolutionary CF explainer wrapping cfts tsevo_cf."""

    def __init__(self, population_size: int = 50, generations: int = 50):
        self.population_size = population_size
        self.generations = generations
        self._dataset: List[Tuple] = []

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        Y_train = classifier.predict(X_train)
        self._dataset = list(zip(X_train, Y_train.tolist()))

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        from cfts.cf_tsevo.tsevo import tsevo_cf

        result = tsevo_cf(
            x,
            self._dataset,
            classifier.model,
            target_class=target_class,
            population_size=self.population_size,
            generations=self.generations,
        )
        if isinstance(result, tuple):
            cf, _ = result
        else:
            cf = result

        if cf is None:
            return x.copy()
        return np.array(cf, dtype=np.float32)
