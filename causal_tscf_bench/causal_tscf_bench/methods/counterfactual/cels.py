"""
M-CELS CF wrapper — multivariate saliency-guided counterfactual.

Wraps cfts_repo's m_cels_generate. Learns a saliency mask that interpolates
between the factual and a nearest-unlike-neighbor (NUN) to produce a CF with
high validity and temporal smoothness.

m_cels_generate expects (M, T) input (channels-first), so we transpose
our (T, M) samples before passing and transpose the result back.

Reference:
  Li et al. (2024), M-CELS: Counterfactual Explanation for Multivariate Time
  Series Data Guided by Learned Saliency Maps. arXiv:2411.02649.
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


class CELSCF(CFExplainer):
    """Saliency-guided CF explainer wrapping cfts m_cels_generate."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        lambda_valid: float = 1.0,
        lambda_sparsity: float = 0.1,
        lambda_smoothness: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_valid = lambda_valid
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smoothness = lambda_smoothness
        self._X_train_MT: np.ndarray | None = None  # (N, M, T)
        self._Y_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        # m_cels expects (N, M, T); X_train is (N, T, M)
        self._X_train_MT = X_train.transpose(0, 2, 1)
        self._Y_train = classifier.predict(X_train)

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        from cfts.cf_cels.cels import m_cels_generate

        # m_cels expects sample as (M, T) channels-first
        x_mt = x.T  # (M, T)

        result = m_cels_generate(
            x_mt,
            classifier.model,
            self._X_train_MT,
            self._Y_train,
            target=target_class,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            lambda_valid=self.lambda_valid,
            lambda_sparsity=self.lambda_sparsity,
            lambda_smoothness=self.lambda_smoothness,
            verbose=False,
        )

        if result is None or (isinstance(result, tuple) and result[0] is None):
            return x.copy()

        cf_raw = result[0] if isinstance(result, tuple) else result
        cf_arr = np.array(cf_raw, dtype=np.float32)

        # cf_raw shape: (1, M, T) or (M, T) — transpose back to (T, M)
        if cf_arr.ndim == 3:
            cf_arr = cf_arr[0]  # → (M, T)
        return cf_arr.T  # → (T, M)
