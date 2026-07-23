"""
CBM-T — Temporal Concept Bottleneck Model wrapper.

Concepts are learned as linear probes over intermediate classifier representations,
one concept per causal variable. Concept activation vectors (CAVs) are computed
per time step, enabling temporal attribution via Axis A (ICC, MCC).

Reference:
  Koh et al. (2020), Concept Bottleneck Models. ICML.
  Temporal extension: aligns concept axes with DAG channels.

Ported from causal_tscf_bench/methods/concept/cbm_t.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ...classifiers.base import TSClassifier
from ..base import AttributionMethod


class CBMT(AttributionMethod):
    """
    Temporal CBM: trains one logistic-regression concept probe per causal channel
    using the channel's time series as the supervision signal.

    Usage
    -----
    1. Call fit_concepts(X_train, concept_labels) with binary concept labels (N, k).
    2. Call attribute(x, classifier) to get (T, k) concept activation map.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 200):
        self.C = C
        self.max_iter = max_iter
        self._probes: list[LogisticRegression] = []
        self._n_concepts: int = 0

    def fit_concepts(self, X_train: np.ndarray, concept_labels: np.ndarray) -> None:
        """Train one logistic probe per causal channel.

        Parameters
        ----------
        X_train : (N, T, k)
        concept_labels : (N, k) — binary concept activation per channel (0/1)
        """
        _, _, k = X_train.shape
        self._n_concepts = k
        self._probes = []
        for m in range(k):
            feats = np.stack([
                X_train[:, :, m].mean(axis=1),
                X_train[:, :, m].std(axis=1),
            ], axis=1)  # (N, 2)
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
            clf.fit(feats, concept_labels[:, m].astype(int))
            self._probes.append(clf)

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        """Return (T, k) concept attribution — uniform over time per concept.

        Each column m is filled with P(concept_m = 1 | x), giving a
        time-uniform activation strength per causal channel.
        """
        T, k = x.shape
        phi = np.zeros((T, k), dtype=np.float64)
        if not self._probes:
            return phi
        for m, probe in enumerate(self._probes):
            feats = np.array([[x[:, m].mean(), x[:, m].std()]])
            proba = probe.predict_proba(feats)[0]
            phi[:, m] = proba[1]  # P(concept=1)
        return phi
