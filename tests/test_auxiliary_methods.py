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
import pytest

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


class TestGraphQualitySweepFracVacuous:
    """M4e: frac_vacuous per sweep point (RISK-17) -- a degraded graph that
    makes the inferred intervention collapse to no-op is a different failure
    from one that mispropagates, and must not hide inside a low graph_error."""

    def _make_scm(self, k=4, L=1, T=20, N=10, seed=0):
        from causaltemp_xai.benchmarks.generator import NlinearSCMT

        gen = NlinearSCMT(k=k, L=L, T=T, N=N, seed=seed, hidden=8)
        data = gen.generate(burn_in=20)
        return data["X"], data["graph"], data["mechanism"]

    def test_frac_vacuous_present_and_in_unit_interval(self):
        from causaltemp_xai.metrics.cf_faith import CFfaith

        _graph_quality_sweep = _phase07._graph_quality_sweep
        build_oracle_interventions = _phase07.build_oracle_interventions

        X, graph, mech = self._make_scm()
        X_sel = X[:5]
        oracle_ints = build_oracle_interventions(X_sel, mech)
        rollout = CFfaith(semantics="noiseless_rollout")
        cf_faith_gt = 1.0  # oracle is faithful by construction

        rows = _graph_quality_sweep(
            graph,
            mech,
            X_sel,
            oracle_ints,
            rollout,
            cf_faith_gt,
            method_adj=(graph > 0).astype(int),
            method_auc=1.0,
            method_label="anchor",
            seed=0,
        )
        assert all("frac_vacuous" in r for r in rows)
        assert all(0.0 <= r["frac_vacuous"] <= 1.0 for r in rows)
        # The true graph (corrupt_frac=0) restricts the mechanism to exactly
        # its real parents, so the inferred CF must match the oracle's own
        # non-vacuous-by-construction intervention -- zero vacuous instances.
        true_row = next(r for r in rows if r["label"] == "corrupt_frac=0")
        assert true_row["frac_vacuous"] == 0.0


class TestGraphQualitySweepEnsemble:
    """M4f (`DECISIONS.md` 2026-08-06): the uncertainty-aware DYNOTEARS ensemble.

    R6 adversarial gate for `graph_error_ensemble_*`/`mean_pairwise_shd`
    before either appears in any result table: a disagreement case (must show
    non-trivial spread), an agreement case (must not fabricate disagreement
    that isn't there), and a degenerate regression anchor (identical members
    must collapse to exactly zero spread).
    """

    def _make_scm(self, k=4, L=1, T=20, N=10, seed=0):
        from causaltemp_xai.benchmarks.generator import NlinearSCMT

        gen = NlinearSCMT(k=k, L=L, T=T, N=N, seed=seed, hidden=8)
        data = gen.generate(burn_in=20)
        return data["X"], data["graph"], data["mechanism"]

    def _sweep_ensemble_args(self, X, graph, mech, n_cf=5):
        from causaltemp_xai.metrics.cf_faith import CFfaith

        build_oracle_interventions = _phase07.build_oracle_interventions
        X_sel = X[:n_cf]
        oracle_ints = build_oracle_interventions(X_sel, mech)
        rollout = CFfaith(semantics="noiseless_rollout")
        cf_faith_gt = 1.0  # oracle is faithful by construction
        return graph, mech, X_sel, oracle_ints, rollout, cf_faith_gt

    # -- surgical isolation: hand-built ensembles, no DYNOTEARS cost --------

    def test_identical_resamples_give_zero_spread(self):
        """B members built from the *same* graph must show exactly zero
        disagreement -- the degenerate regression anchor."""
        _graph_quality_sweep_ensemble = _phase07._graph_quality_sweep_ensemble

        X, graph, mech = self._make_scm()
        args = self._sweep_ensemble_args(X, graph, mech)
        true_bin = (graph > 0).astype(int)
        ensemble_adjs = [true_bin.copy() for _ in range(5)]

        result = _graph_quality_sweep_ensemble(
            *args, ensemble_adjs, method_label="dynotears", seed=0
        )

        assert result["mean_pairwise_shd"] == 0.0
        ge = result["graph_error_ensemble"]
        assert ge["ci_lo"] == pytest.approx(ge["mean"])
        assert ge["ci_hi"] == pytest.approx(ge["mean"])
        assert result["ensemble_b"] == 5
        assert len(result["ensemble_anchor_points"]) == 5

    def test_disagreeing_hand_built_ensemble_shows_nonzero_spread(self):
        """Half the ensemble is the true graph, half is fully corrupted
        (`_corrupt_graph(frac=1.0)`) -- must show large, not zero, spread.
        Proves the spread metric tracks real instability by construction,
        independent of DYNOTEARS's own fitting noise."""
        _graph_quality_sweep_ensemble = _phase07._graph_quality_sweep_ensemble
        _corrupt_graph = _phase07._corrupt_graph

        X, graph, mech = self._make_scm()
        args = self._sweep_ensemble_args(X, graph, mech)
        true_bin = (graph > 0).astype(int)
        rng = np.random.default_rng(0)
        corrupted = _corrupt_graph(true_bin, frac=1.0, rng=rng)
        ensemble_adjs = [true_bin.copy(), true_bin.copy(), corrupted, corrupted]

        result = _graph_quality_sweep_ensemble(
            *args, ensemble_adjs, method_label="dynotears", seed=0
        )

        assert result["mean_pairwise_shd"] > 0.0
        ge = result["graph_error_ensemble"]
        assert ge["ci_hi"] - ge["ci_lo"] > 0.0

    def test_pairwise_shd_matches_direct_computation(self):
        """`mean_pairwise_shd` is the mean SHD across all C(B,2) pairs of the
        ensemble's own graphs (not against the true graph) -- verify against
        a direct, independent computation."""
        from causaltemp_xai.metrics.axis_a import shd

        _graph_quality_sweep_ensemble = _phase07._graph_quality_sweep_ensemble

        X, graph, mech = self._make_scm()
        args = self._sweep_ensemble_args(X, graph, mech)
        true_bin = (graph > 0).astype(int)
        rng = np.random.default_rng(1)
        _corrupt_graph = _phase07._corrupt_graph
        ensemble_adjs = [
            true_bin.copy(),
            _corrupt_graph(true_bin, frac=0.5, rng=rng),
            _corrupt_graph(true_bin, frac=1.0, rng=rng),
        ]

        result = _graph_quality_sweep_ensemble(
            *args, ensemble_adjs, method_label="dynotears", seed=0
        )

        expected = np.mean(
            [
                shd(ensemble_adjs[0], ensemble_adjs[1]),
                shd(ensemble_adjs[0], ensemble_adjs[2]),
                shd(ensemble_adjs[1], ensemble_adjs[2]),
            ]
        )
        assert result["mean_pairwise_shd"] == pytest.approx(expected)

    # -- real end-to-end: _fit_graph_method_ensemble + DYNOTEARS refits -----

    def test_fit_dynotears_ensemble_produces_b_distinct_shaped_graphs(self):
        """Real bootstrap-resample-and-refit path: B independent DYNOTEARS
        fits, each a valid density-matched adjacency of the right shape."""
        _fit_graph_method_ensemble = _phase07._fit_graph_method_ensemble

        X, graph, _mech = self._make_scm(k=4, L=1, T=20, N=60, seed=0)
        k = graph.shape[0]
        n_true = int((graph > 0).sum())

        members = _fit_graph_method_ensemble(X, k=k, L=1, n_true=n_true, B=3, seed=0)

        assert len(members) == 3
        for adj in members:
            assert adj.shape == graph.shape
            assert adj.dtype == int
            assert adj.sum() == n_true  # density-matched: same edge count every time

    def test_fit_dynotears_ensemble_deterministic_given_seed(self):
        _fit_graph_method_ensemble = _phase07._fit_graph_method_ensemble

        X, graph, _mech = self._make_scm(k=4, L=1, T=20, N=60, seed=0)
        k = graph.shape[0]
        n_true = int((graph > 0).sum())

        m1 = _fit_graph_method_ensemble(X, k=k, L=1, n_true=n_true, B=3, seed=7)
        m2 = _fit_graph_method_ensemble(X, k=k, L=1, n_true=n_true, B=3, seed=7)

        for a, b in zip(m1, m2):
            assert np.array_equal(a, b)

    def test_end_to_end_ensemble_self_consistent(self):
        """Real DYNOTEARS-ensemble + sweep-aggregation, wired together end to
        end. Not asserting a specific spread value (real fits are noisy at
        test scale) -- only that the pipeline runs and every invariant the
        surgical tests above establish by construction also holds for real
        output: valid CI ordering, non-negative spread, correct B."""
        _fit_graph_method_ensemble = _phase07._fit_graph_method_ensemble
        _graph_quality_sweep_ensemble = _phase07._graph_quality_sweep_ensemble

        X, graph, mech = self._make_scm(k=4, L=1, T=20, N=80, seed=0)
        k = graph.shape[0]
        n_true = int((graph > 0).sum())
        args = self._sweep_ensemble_args(X, graph, mech)

        members = _fit_graph_method_ensemble(X, k=k, L=1, n_true=n_true, B=3, seed=0)
        result = _graph_quality_sweep_ensemble(*args, members, method_label="dynotears", seed=0)

        assert result["ensemble_b"] == 3
        assert result["mean_pairwise_shd"] >= 0.0
        ge = result["graph_error_ensemble"]
        assert ge["ci_lo"] <= ge["mean"] <= ge["ci_hi"]
        assert len(result["ensemble_anchor_points"]) == 3
