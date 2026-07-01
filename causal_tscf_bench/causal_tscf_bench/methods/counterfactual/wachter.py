"""
Wachter CF wrapper — gradient-based closest CF search.

Wraps cfts_repo's wachter_genetic_cf as the tabular/baseline method
replacing DiCE. No dataset needed; optimizes directly via gradient steps.

Reference:
  Wachter et al. (2017), Counterfactual Explanations without Opening the
  Black Box. arXiv:1711.00399.
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


class WachterCF(CFExplainer):
    """Gradient-based counterfactual via cfts wachter_genetic_cf."""

    def __init__(self, max_steps: int = 500, step_size: float = 0.1):
        self.max_steps = max_steps
        self.step_size = step_size

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        pass  # no training needed

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        from cfts.cf_wachter.wachter import wachter_genetic_cf

        result = wachter_genetic_cf(
            x,
            classifier.model,
            target=target_class,
            max_steps=self.max_steps,
            step_size=self.step_size,
            verbose=False,
        )
        # wachter_genetic_cf may return (cf, scores) or just cf depending on version
        if isinstance(result, tuple):
            cf, _ = result
        else:
            cf = result

        if cf is None:
            return x.copy()
        return np.array(cf, dtype=np.float32)
