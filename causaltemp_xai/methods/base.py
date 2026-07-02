"""Abstract interfaces for all explainer families.

CFExplainer: produces a counterfactual trajectory X_cf given x and target class.
AttributionMethod: produces a feature-time importance map phi(x) of shape (T, k).

These interfaces are the contract used by the evaluation harness (Axis C / A / D).

Ported from causal_tscf_bench/methods/base.py and adapted to causaltemp_xai
conventions ((N, T, k) shape, LSTMClassifier as the concrete classifier).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CFExplainer(ABC):
    """Abstract counterfactual explainer.

    Existing methods (WachterCF, DiCECF, CARLARecourse) expose a ``generate()``
    interface.  The ``fit()`` / ``explain()`` methods below are aliases added for
    compatibility with the bench's evaluation harness.  Subclasses may implement
    either interface; ``generate()`` remains the primary API.
    """

    def fit(self, X_train: np.ndarray, classifier) -> None:
        """Fit any method-specific structures (e.g., nearest-neighbor index).

        Default implementation is a no-op — override when needed.
        """
        pass

    @abstractmethod
    def explain(
        self,
        x: np.ndarray,
        target_class: int,
        classifier,
    ) -> np.ndarray:
        """Generate a single counterfactual for instance x.

        Parameters
        ----------
        x : np.ndarray
            Shape (T, k) — single factual instance.
        target_class : int
            Desired output class y*.
        classifier : LSTMClassifier or TSClassifier

        Returns
        -------
        np.ndarray
            X_cf of shape (T, k).
        """
        ...

    def explain_batch(
        self,
        X: np.ndarray,
        target_classes: np.ndarray,
        classifier,
    ) -> np.ndarray:
        """Default batch: loop over instances. Override for efficiency."""
        results = []
        for x, y in zip(X, target_classes):
            results.append(self.explain(x, int(y), classifier))
        return np.stack(results, axis=0)


class AttributionMethod(ABC):
    """Abstract attribution / saliency method."""

    @abstractmethod
    def attribute(
        self,
        x: np.ndarray,
        classifier,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute feature-time importance map.

        Parameters
        ----------
        x : np.ndarray
            Shape (T, k).
        classifier : LSTMClassifier or TSClassifier
        target_class : int | None
            Class to attribute toward. Defaults to the predicted class.

        Returns
        -------
        np.ndarray
            Shape (T, k) — signed or unsigned importance scores.
        """
        ...
