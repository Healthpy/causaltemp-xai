"""Adversarial tests for Axis-B robustness metrics (M1, objective O1).

The guarded environment-shift ``shift_vr`` (validity retention for CF methods)
is covered in ``tests/test_metric_adversarial.py``. This module covers the two
*attribution/concept* robustness metrics that had no adversarial coverage:

* ``input_sensitivity`` — must score a jittery attributor (large output swing
  under a tiny input perturbation) *above* a smooth one, and score an
  input-independent attributor at exactly zero.
* ``concept_stability`` — must score an instance-varying concept function
  *above* a constant one, which sits at zero variance.

Contract per metric: reject the constructed unstable case, reward the stable
one. Governed by R6.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.metrics.axis_b import concept_stability, input_sensitivity

# ---------------------------------------------------------------------------
# input_sensitivity — attribution stability under small input perturbations
# ---------------------------------------------------------------------------


class TestInputSensitivityAdversarial:
    def _X(self):
        # Offset away from the origin so the relative-L2 denominator is well
        # conditioned for the smooth (identity) attributor.
        return np.random.default_rng(0).normal(size=(4, 10, 3)) + 5.0

    def test_flags_jittery_attribution(self):
        X = self._X()
        smooth = lambda x: x.copy()  # noqa: E731 — Lipschitz-1 in the input
        jittery = lambda x: np.sin(1000.0 * x)  # noqa: E731 — huge swing for tiny delta
        s_smooth = input_sensitivity(X, smooth, eps=0.01, n_trials=8, rng=np.random.default_rng(2))
        s_jittery = input_sensitivity(
            X, jittery, eps=0.01, n_trials=8, rng=np.random.default_rng(2)
        )
        assert s_jittery > s_smooth

    def test_input_independent_attribution_is_insensitive(self):
        X = self._X()
        const = lambda x: np.ones_like(x)  # noqa: E731
        s = input_sensitivity(X, const, eps=0.05, n_trials=5, rng=np.random.default_rng(0))
        assert s == 0.0

    def test_nonnegative(self):
        X = self._X()
        s = input_sensitivity(
            X, lambda x: x.copy(), eps=0.01, n_trials=5, rng=np.random.default_rng(3)
        )
        assert s >= 0.0


# ---------------------------------------------------------------------------
# concept_stability — variance of concept activations across a group
# ---------------------------------------------------------------------------


class TestConceptStabilityAdversarial:
    def _group(self):
        return np.random.default_rng(0).normal(size=(20, 10, 3))

    def test_constant_concept_is_perfectly_stable(self):
        Xg = self._group()
        const_fn = lambda x: np.array([1.0, 2.0, 3.0])  # noqa: E731 — ignores instance
        assert concept_stability(const_fn, Xg) == 0.0

    def test_flags_instance_varying_concept(self):
        Xg = self._group()
        varying_fn = lambda x: x.mean(axis=0)  # noqa: E731 — differs per instance
        assert concept_stability(varying_fn, Xg) > 0.0

    def test_varying_exceeds_constant(self):
        Xg = self._group()
        const_fn = lambda x: np.array([1.0, 2.0, 3.0])  # noqa: E731
        varying_fn = lambda x: x.mean(axis=0)  # noqa: E731
        assert concept_stability(varying_fn, Xg) > concept_stability(const_fn, Xg)
