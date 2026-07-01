"""
CounTS CF wrapper — causal VAE-based counterfactual.

Trains a CounTS VAE (encoder/decoder/predictor) on the training data in fit(),
then generates counterfactuals by optimizing in latent space via the causal
ladder (Abduction → Action → Prediction).

Reference:
  Delaney et al. (2024), CounTS: Causal Counterfactual Explanations for
  Time Series via Structural Causal Models.
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


class CountsCF(CFExplainer):
    """VAE-based causal CF explainer wrapping cfts CounTS."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        train_epochs: int = 50,
        max_iter: int = 300,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.train_epochs = train_epochs
        self.max_iter = max_iter
        self._counts_model = None

    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        from cfts.cf_counts.counts import CounTSModel, train_counts_model

        Y_train = classifier.predict(X_train)
        dataset: List[Tuple] = list(zip(X_train, Y_train.tolist()))

        N, T, M = X_train.shape
        num_classes = len(np.unique(Y_train))

        import torch
        device = next(classifier.model.parameters()).device
        model = CounTSModel(
            input_dim=M,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_classes=num_classes,
            seq_len=T,
        ).to(device)

        train_counts_model(model, dataset, num_epochs=self.train_epochs, verbose=False)
        model.eval()
        self._counts_model = model

    def explain(self, x: np.ndarray, target_class: int, classifier: TSClassifier) -> np.ndarray:
        if self._counts_model is None:
            raise RuntimeError("Call fit() before explain().")

        from cfts.cf_counts.counts import counts_generate_counterfactual

        # counts_generate_counterfactual returns (x_cf, y_cf) in same shape as x
        result = counts_generate_counterfactual(
            x,
            self._counts_model,
            target_class=target_class,
            max_iter=self.max_iter,
        )
        if isinstance(result, tuple):
            cf, _ = result
        else:
            cf = result

        if cf is None:
            return x.copy()
        cf_arr = np.array(cf, dtype=np.float32)
        # counts_generate_counterfactual may flip orientation due to an internal
        # correction heuristic — ensure the result matches x's (T, M) shape.
        if cf_arr.shape != x.shape and cf_arr.T.shape == x.shape:
            cf_arr = cf_arr.T
        return cf_arr
