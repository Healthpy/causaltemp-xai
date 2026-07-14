"""Official TimeSHAP attribution (Bento et al., 2021, KDD).

This module wraps the *official* ``timeshap`` library (feedzai/timeshap on
PyPI) behind this codebase's :class:`AttributionMethod` contract, producing a
dense ``(T, k)`` feature-time importance map.

This replaces the earlier ``MCMaskSHAP`` proxy (a flat Monte-Carlo
random-coalition sampler that was *not* TimeSHAP and shipped under a disclosure
note per docs/PROJECT_PLAN.md Standing Decision #3 / risk R4). Because this is
now the genuine method, no proxy disclosure is required.

Reference: Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., &
Bizarro, P. (2021). TimeSHAP: Explaining Recurrent Models through Sequence
Perturbations. KDD 2021. Official implementation: ``timeshap`` on PyPI.

Adapter design notes
--------------------
TimeSHAP's native workflow is pruning -> event-level -> feature-level ->
cell-level Shapley estimation. To satisfy the ``attribute() -> (T, k)``
contract (a dense saliency grid rather than TimeSHAP's usual top-k sparse
cell report) this adapter:

- **Bypasses the pruning stage** (``pruned_idx = 0``): every timestep is kept
  as its own coalition member so the grid covers the whole sequence. TimeSHAP's
  event-display math misbehaves for negative ``pruned_idx`` (it emits
  ``T - pruned_idx`` event rows), so a non-negative index is required here.
- Requests **all** cells (``top_x_events = T``, ``top_x_feats = k``) at the
  cell level, then pivots the returned ``(Event, Feature, Shapley Value)``
  table into a ``(T, k)`` array. TimeSHAP labels the most recent timestep
  ``Event -1``; this maps to grid index ``T - 1`` (``Event -i`` -> ``T - i``).
  Aggregate rows ("Other Events" / "Other Features") are skipped.
- Uses the **average event** as the perturbation baseline (TimeSHAP's own
  convention), computed from the background set passed to :meth:`fit`. If
  :meth:`fit` is not called, a zero average event is used.
"""

from __future__ import annotations

import numpy as np

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier


class TimeSHAP(AttributionMethod):
    """Official TimeSHAP cell-level Shapley attribution as a dense ``(T, k)`` map.

    Parameters
    ----------
    nsamples : int, default 200
        Number of coalitions TimeSHAP samples at each level.
    seed : int, default 0
        Random seed threaded through TimeSHAP's kernel sampler. Fixed by
        default so repeated calls are reproducible, matching the rest of this
        codebase (``LinearSCMT``, ``LSTMClassifier``, ``stratified_split``, ...).
    """

    def __init__(self, nsamples: int = 200, seed: int = 0):
        self.nsamples = nsamples
        self.seed = seed
        self._baseline_event: np.ndarray | None = None  # (1, k) average event

    def fit(self, X_train: np.ndarray) -> None:
        """Compute the average-event baseline from a background set ``(N, T, k)``."""
        import pandas as pd
        from timeshap.utils import calc_avg_event

        Xt = np.asarray(X_train, dtype=float)
        k = Xt.shape[-1]
        cols = [str(i) for i in range(k)]
        bg = pd.DataFrame(Xt.reshape(-1, k), columns=cols)
        self._baseline_event = calc_avg_event(bg, numerical_feats=cols, categorical_feats=[]).values

    def attribute(
        self, x: np.ndarray, classifier: TSClassifier, target_class: int | None = None
    ) -> np.ndarray:
        from timeshap.explainer import local_cell_level, local_event, local_feat

        x = np.asarray(x, dtype=np.float32)
        T, k = x.shape
        if target_class is None:
            target_class = int(classifier.predict(x[np.newaxis])[0])

        baseline = (
            self._baseline_event
            if self._baseline_event is not None
            else np.zeros((1, k), dtype=np.float64)
        )
        cols = [str(i) for i in range(k)]

        def f(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            return classifier.predict_proba(arr)[:, target_class : target_class + 1].astype(
                np.float32
            )

        data = x[np.newaxis]  # (1, T, k)
        pruned_idx = 0  # no pruning: keep every timestep (see module docstring)
        ev = local_event(
            f,
            data,
            {"rs": self.seed, "nsamples": self.nsamples},
            "0",
            "id",
            baseline,
            pruned_idx,
        )
        ft = local_feat(
            f,
            data,
            {"rs": self.seed, "nsamples": self.nsamples, "feature_names": cols},
            "0",
            "id",
            baseline,
            pruned_idx,
        )
        cell = local_cell_level(
            f,
            data,
            {"top_x_events": T, "top_x_feats": k, "rs": self.seed, "nsamples": self.nsamples},
            ev,
            ft,
            "0",
            "id",
            baseline,
            pruned_idx,
        )

        phi = np.zeros((T, k), dtype=np.float64)
        for _, row in cell.iterrows():
            event = str(row["Event"])
            feat = str(row["Feature"])
            # Skip aggregate rows ("Other Events" / "Other Features").
            if not event.startswith("Event ") or feat not in cols:
                continue
            e = int(event.split()[-1])  # negative index; "Event -1" -> last step
            t = T + e
            if 0 <= t < T:
                phi[t, int(feat)] = float(row["Shapley Value"])
        return phi
