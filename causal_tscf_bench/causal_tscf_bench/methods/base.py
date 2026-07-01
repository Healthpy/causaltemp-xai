"""
Abstract interfaces for all explainer families.

CFExplainer: produces a counterfactual trajectory X'_exp given x and target class.
AttributionMethod: produces a feature-time importance map phi(x) of shape (T, M).

These interfaces are the contract used by the evaluation harness (Axis C / A / D).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..classifiers.base import TSClassifier


class CFExplainer(ABC):
    """Abstract counterfactual explainer."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, classifier: TSClassifier) -> None:
        """Fit any method-specific structures (e.g., nearest-neighbor index, VAE)."""
        ...

    @abstractmethod
    def explain(
        self,
        x: np.ndarray,
        target_class: int,
        classifier: TSClassifier,
    ) -> np.ndarray:
        """
        Generate a single counterfactual for instance x.

        Parameters
        ----------
        x : np.ndarray
            Shape (T, M) — single factual instance.
        target_class : int
            Desired output class y*.
        classifier : TSClassifier

        Returns
        -------
        np.ndarray
            X'_exp of shape (T, M).
        """
        ...

    def explain_batch(
        self,
        X: np.ndarray,
        target_classes: np.ndarray,
        classifier: TSClassifier,
    ) -> np.ndarray:
        """Default batch: loop over instances. Override for efficiency."""
        results = []
        for i, (x, y) in enumerate(zip(X, target_classes)):
            results.append(self.explain(x, int(y), classifier))
        return np.stack(results, axis=0)


class AttributionMethod(ABC):
    """Abstract attribution / saliency method."""

    @abstractmethod
    def attribute(
        self,
        x: np.ndarray,
        classifier: TSClassifier,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute feature-time importance map.

        Parameters
        ----------
        x : np.ndarray
            Shape (T, M).
        classifier : TSClassifier
        target_class : int | None
            Class to attribute toward. Defaults to the predicted class.

        Returns
        -------
        np.ndarray
            Shape (T, M) — signed or unsigned importance scores.
        """
        ...
