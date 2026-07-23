"""ChannelConceptProbe — per-channel logistic concept probe.

**Naming note (R3 remediation, see docs/method_provenance.md):** this class was
previously named ``CBMT`` ("Temporal Concept Bottleneck Model") with a
docstring citing Koh et al. (2020), *Concept Bottleneck Models*, ICML. It does
not implement that architecture: a real CBM inserts a bottleneck of concept
activations *between* a shared feature extractor and the task predictor, and
trains the concept probes and the downstream classifier jointly (or
sequentially) so the concepts are causally load-bearing for the prediction.
This class does neither — it fits one independent logistic-regression probe
per causal channel directly on two hand-picked summary statistics (per-channel
mean and std over the whole window), with no coupling to the classifier being
explained at all (the ``classifier`` argument to :meth:`attribute` is accepted
for interface compatibility and unused), and it returns a *time-uniform* map
(same value repeated across all ``T`` steps), discarding temporal structure
entirely. Calling that "CBM-T" claimed fidelity to a published method it does
not implement. Renamed to describe what it actually is: an independent
per-channel concept probe over simple summary statistics.

Reference (for the concept-probe idea in general, not implemented here):
  Koh et al. (2020), Concept Bottleneck Models. ICML.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ...classifiers.base import TSClassifier
from ..base import AttributionMethod


class ChannelConceptProbe(AttributionMethod):
    """Per-channel logistic concept probe (not a concept bottleneck model).

    Trains one independent logistic-regression probe per causal channel using
    that channel's own (mean, std) summary statistics as features and a
    user-supplied binary concept label as supervision. At attribution time,
    each channel's map is filled uniformly across time with
    ``P(concept_m = 1 | x)`` — a time-uniform, per-channel concept-activation
    strength. There is no bottleneck layer and no coupling with a downstream
    classifier's representations or predictions.

    Usage
    -----
    1. Call ``fit_concepts(X_train, concept_labels)`` with binary concept
       labels ``(N, k)``.
    2. Call ``attribute(x, classifier)`` to get a ``(T, k)`` concept map.
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
            feats = np.stack(
                [
                    X_train[:, :, m].mean(axis=1),
                    X_train[:, :, m].std(axis=1),
                ],
                axis=1,
            )  # (N, 2)
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
            clf.fit(feats, concept_labels[:, m].astype(int))
            self._probes.append(clf)

    def attribute(
        self, x: np.ndarray, classifier: TSClassifier, target_class: int | None = None
    ) -> np.ndarray:
        """Return (T, k) concept attribution — uniform over time per concept.

        Each column m is filled with P(concept_m = 1 | x), giving a
        time-uniform activation strength per causal channel. ``classifier``
        and ``target_class`` are accepted for interface compatibility with
        :class:`~causaltemp_xai.methods.base.AttributionMethod` and are not
        used — this probe is fit independently of any classifier.
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
