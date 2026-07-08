"""Monte-Carlo masking SHAP approximation — a random-coalition Shapley estimator.

This module implements a crude Monte-Carlo estimator of Shapley values over
(time, feature) coordinates: it draws random binary masks ("coalitions") over
the ``(T, k)`` grid, includes/excludes one target coordinate from each pair
of masks, and averages the resulting change in predicted-class probability.
This is a generic random-coalition Shapley sampler. It is NOT an
implementation of Bento et al.'s TimeSHAP.

Naming history / disclosure
----------------------------
An earlier revision of this codebase named this class ``TimeSHAP``. That
naming violated this lab's standing decision that no proxy implementation
ships under an original method's name (docs/PROJECT_PLAN.md, Standing
Decision #3) and is tracked as risk R4 ("misrepresented baselines...fatal if
published"). Compounding the naming problem, the previous implementation's
random-number generator was constructed with ``np.random.default_rng()``
(no seed argument at all), inside ``attribute()``, so two calls with
identical inputs could return different attributions -- a reproducibility
bug independent of, but aggravating, the naming problem. Both issues are
fixed here: the class is renamed to ``MCMaskSHAP`` ("Monte-Carlo masking
SHAP approximation") and its RNG is now seeded and stored as an instance
attribute (see ``seed`` below). No change was made to the sampling
algorithm itself -- only the name, docstrings, and seeding.

What this is NOT (relationship to the real TimeSHAP)
-----------------------------------------------------
TimeSHAP (Bento et al., 2021, "TimeSHAP: Explaining Recurrent Models through
Sequence Perturbations", KDD 2021) computes Shapley values with a
*structured* pruning and coalition scheme over an explicit event/feature/cell
hierarchy: it first prunes long sequences to the event horizon that matters
(via a Shapley-value-based pruning algorithm over "events" -- contiguous
time windows), then estimates feature-level Shapley values, then optionally
cell-level (event x feature) Shapley values, each level's coalition sampling
informed by the coarser level above it. That hierarchical
pruning-then-refining structure, and TimeSHAP's specific perturbation
operator (replacing masked-out entries with a learned or empirical
"average event" rather than an arbitrary baseline), are not implemented
here. ``MCMaskSHAP`` instead draws i.i.d. uniformly random binary masks
directly over the full ``(T, k)`` grid with no pruning stage and no
hierarchy -- a flat, unstructured Monte-Carlo Shapley sampler in the general
family KernelSHAP/permutation-SHAP belong to, not TimeSHAP's method.

The reference below is retained only as "the method this class was
originally intended to approximate" -- it is not a description of this
class's identity or an implementation source.

Reference (approximation target, NOT an implementation source): Bento, J.,
Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., & Bizarro, P. (2021).
TimeSHAP: Explaining Recurrent Models through Sequence Perturbations. KDD
2021.

An official implementation exists (``feedzai/timeshap`` on GitHub/PyPI as
``timeshap``); see docs/method_provenance.md for a note on what integrating
it would require instead of this proxy.

Ported from causal_tscf_bench/methods/attribution/timeshap.py (file renamed
from timeshap.py to mc_mask_shap.py as part of this disclosure).
"""

from __future__ import annotations

import numpy as np

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier


class MCMaskSHAP(AttributionMethod):
    """Monte-Carlo random-coalition Shapley-value approximation.

    For ``n_samples`` iterations, draws a uniformly random binary mask over
    the ``(T, k)`` grid, picks one target coordinate ``(t, m)``, forces it
    ``True`` in one copy of the mask and ``False`` in another, and attributes
    the resulting change in the target class's predicted probability to
    ``(t, m)``. Masked-out entries are replaced by ``baseline`` (zeros, or
    the per-coordinate training-set mean when ``baseline="mean"`` and
    :meth:`fit` was called). The per-coordinate attributions are averaged
    over the ``n_samples`` draws that touched each coordinate's role.

    This is a flat, unstructured Monte-Carlo Shapley sampler -- it has no
    event-level pruning stage and no event/feature/cell hierarchy, so it is
    NOT TimeSHAP (Bento et al., 2021). See the module docstring for the full
    disclosure.

    Parameters
    ----------
    n_samples : int, default 200
        Number of random-coalition draws.
    baseline : {"zero", "mean"}, default "zero"
        Perturbation baseline for masked-out entries. ``"mean"`` requires
        calling :meth:`fit` first.
    seed : int | None, default None
        Seed for the internal ``numpy.random.Generator``. When ``None``,
        the generator falls back to a **fixed internal seed of 0**, so
        ``MCMaskSHAP()`` (no explicit seed) is deterministic and repeated
        calls with the same inputs return bit-identical output -- this
        matches the convention used elsewhere in this codebase, where every
        other seeded stochastic component (``LinearSCMT``, ``NlinearSCMT``,
        ``LSTMClassifier``, ``stratified_split``, the bootstrap-CI helpers in
        ``causaltemp_xai/stats.py``, and every ``BenchmarkConfig`` preset in
        ``causaltemp_xai/config.py``) defaults to a fixed explicit integer
        seed rather than to true nondeterminism. ``seed=None`` here is
        therefore "use the codebase default seed," not "be nondeterministic"
        -- callers who want a fresh random draw per call must pass an
        explicit varying seed (e.g. drawn from their own seeded RNG)
        themselves. This differs from raw ``numpy.random.default_rng(None)``
        semantics (OS-entropy seeding) by design, to keep this method
        reproducible-by-default like the rest of the pipeline.
    """

    def __init__(self, n_samples: int = 200, baseline: str = "zero",
                 seed: int | None = None):
        self.n_samples = n_samples
        self.baseline = baseline   # "zero" or "mean"
        self.seed = seed
        self._baseline_val: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> None:
        if self.baseline == "mean":
            self._baseline_val = X_train.mean(axis=0)  # (T, k)

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        T, k = x.shape
        if target_class is None:
            target_class = int(classifier.predict(x[np.newaxis])[0])

        baseline = (self._baseline_val if self._baseline_val is not None
                    else np.zeros_like(x))
        # seed=None -> fixed internal default (0), for reproducibility by
        # default; see the class docstring for the rationale.
        rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        phi = np.zeros((T, k), dtype=np.float64)

        for _ in range(self.n_samples):
            t_feat = rng.integers(T)
            m_feat = rng.integers(k)

            mask_with = rng.integers(0, 2, size=(T, k)).astype(bool)
            mask_with[t_feat, m_feat] = True
            mask_without = mask_with.copy()
            mask_without[t_feat, m_feat] = False

            x_with = np.where(mask_with, x, baseline)
            x_without = np.where(mask_without, x, baseline)

            p_with = classifier.predict_proba(x_with[np.newaxis])[0, target_class]
            p_without = classifier.predict_proba(x_without[np.newaxis])[0, target_class]
            phi[t_feat, m_feat] += (p_with - p_without)

        phi /= self.n_samples
        return phi
