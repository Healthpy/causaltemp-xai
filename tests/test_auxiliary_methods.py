"""Tests for Phase 07's per-method propagation-error helper.

Found 2026-08-04 while backing H3a's cited graph-error number with a
regenerated artifact: ``_mean_soft_cf_faith`` used plain ``np.mean`` over
per-instance CF-faith scores, but ``CFfaith.score`` deliberately returns NaN
under its degeneracy gate (``intervention_t >= T - 1``) -- a rollout window of
zero length has no evidence either way. On ``full_nl``, CftsCels derives
``intervention_t = T - 1`` for roughly 60% of the selected instances, so plain
``np.mean`` turned a real, computable per-method score into a silent NaN for
every method that ever hits the gate. Every other CF-faith aggregator in this
codebase already treats a gated instance as an abstention via ``nanmean``.

Phase 07 is a numbered-prefix module, loaded the way
``tests/test_horizon_sweep.py`` already loads Phase 06.
"""

from __future__ import annotations

import importlib

import numpy as np

_phase07 = importlib.import_module("experiments.07_auxiliary_methods")
_mean_soft_cf_faith = _phase07._mean_soft_cf_faith


class _ConstantMechanism:
    """Predicts the last observed value -- enough to drive CFfaith.score."""

    L = 1

    def forward_numpy(self, window):
        return window[-1]


class TestMeanSoftCfFaith:
    def _run(self, x_sel, cfs):
        graph = np.ones((x_sel.shape[-1], x_sel.shape[-1], 1))
        from causaltemp_xai.metrics.cf_faith import CFfaith

        scorer = CFfaith(semantics="noiseless_rollout")
        return _mean_soft_cf_faith(x_sel, cfs, graph, _ConstantMechanism(), scorer)

    def test_one_gated_instance_does_not_poison_the_mean(self):
        """A CF whose derived intervention lands on T-1 scores NaN by design
        (the degeneracy gate) and must be excluded, not propagated."""
        T, k, n = 6, 2, 3
        rng = np.random.default_rng(0)
        x_sel = rng.normal(size=(n, T, k))
        cfs = x_sel.copy()
        # Instances 0-1: intervention at t=1, plenty of rollout window left --
        # each contributes a real, non-gated score.
        cfs[0, 1] += 1.0
        cfs[1, 1] += 1.0
        # Instance 2: no edit at all before the last step, so
        # derive_intervention_t lands on T-1 -- the degeneracy gate.
        cfs[2, -1] += 1.0

        val = self._run(x_sel, cfs)
        assert not np.isnan(val), "one gated instance must not NaN the whole mean"

        # The gated instance is excluded entirely: the result must equal the
        # mean of instances 0-1 alone, not a value diluted by treating the
        # gate as some other number.
        expected = self._run(x_sel[:2], cfs[:2])
        assert val == expected

    def test_all_gated_returns_nan_not_zero(self):
        """If every instance hits the degeneracy gate there is no evidence at
        all -- the honest result is NaN, not a fabricated 0.0 or 1.0."""
        T, k, n = 5, 2, 2
        rng = np.random.default_rng(1)
        x_sel = rng.normal(size=(n, T, k))
        cfs = x_sel.copy()
        cfs[:, -1] += 1.0  # every instance edits only the last timestep

        val = self._run(x_sel, cfs)
        assert np.isnan(val)

    def test_empty_input_returns_nan(self):
        x_sel = np.zeros((0, 5, 2))
        cfs = np.zeros((0, 5, 2))
        assert np.isnan(self._run(x_sel, cfs))
