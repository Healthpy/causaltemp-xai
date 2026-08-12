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


class TestScaleFreeGraphError:
    """R6 adversarial gate for ``graph_error_sigma`` / ``propagation_error_sigma``
    (M4e re-derivation, 2026-08-12), before either appears in a result table.

    The defect being fixed: ``graph_error`` is a difference of *soft* scores,
    and ``soft = exp(-residual)`` is strongly compressive near ``residual = 0``.
    Two mechanism families that degrade by the *same relative amount* therefore
    post wildly different ``graph_error`` values purely because their states
    live at different magnitudes -- which is exactly the cross-family
    comparison the dissipation claim rests on. The adversarial case is a pure
    rescaling: it changes nothing causal, so a scale-free metric must return
    the identical number, and the old metric must be shown to fail it (else
    the new column buys nothing).
    """

    #: Residual ladder for the "small-state" family: an oracle floor plus four
    #: increasingly corrupted graph points, all in raw state units.
    _RESIDUALS = (0.001, 0.0015, 0.002, 0.0035, 0.005)

    def _ladder(self, c: float):
        """The same ladder with every state magnitude multiplied by ``c``:
        residuals scale by ``c`` (they are L1 distances in state units) and so
        does sigma. Nothing about the causal degradation changes."""
        soft = np.exp(-np.asarray(self._RESIDUALS) * c)
        resid = _phase07.soft_to_residual(soft)
        sigma = 1.0 * c
        r_gt = resid[0]
        graph_error = float(soft[0]) - soft  # the old, exp-compressed column
        graph_error_sigma = (resid - r_gt) / sigma
        return graph_error, graph_error_sigma

    def test_soft_to_residual_is_the_exact_inverse(self):
        r = np.array([1e-6, 0.001, 0.5, 3.0, 20.0])
        assert _phase07.soft_to_residual(np.exp(-r)) == pytest.approx(r, rel=1e-9)

    def test_soft_to_residual_handles_the_gate_and_retro_branches(self):
        """``soft == 0`` is the retroactive-change branch (an infinite penalty
        by construction, not a measurement); NaN is the degeneracy gate. Both
        must survive the inversion as themselves, and ``_mean_residual`` must
        drop both rather than returning inf or NaN for the whole method."""
        assert _phase07.soft_to_residual(0.0) == float("inf")
        assert np.isnan(_phase07.soft_to_residual(float("nan")))
        arr = np.array([np.exp(-0.2), np.exp(-0.4), 0.0, np.nan])
        assert _phase07._mean_residual(arr) == pytest.approx(0.3)
        assert np.isnan(_phase07._mean_residual(np.array([0.0, np.nan])))

    def test_rescaling_leaves_the_scale_free_column_invariant(self):
        """The adversarial case. A 100x state rescaling is causally a no-op."""
        _, sigma_small = self._ladder(1.0)
        _, sigma_large = self._ladder(100.0)
        assert sigma_large == pytest.approx(sigma_small, rel=1e-9)

    def test_the_old_column_fails_the_same_case(self):
        """The fix is only worth its column if the old metric demonstrably
        breaks here. It does: the identical causal degradation reads ~100x
        larger once the states are 100x bigger, because ``exp`` is no longer
        near-linear at that residual scale."""
        raw_small, _ = self._ladder(1.0)
        raw_large, _ = self._ladder(100.0)
        assert raw_large[-1] / raw_small[-1] > 50.0

    def test_sweep_reports_a_scale_free_column_that_is_zero_at_the_true_graph(self):
        """End-to-end through the real sweep: the column exists, the true graph
        sits at exactly 0 (it is the normalisation's own baseline), and no
        point is spuriously negative beyond float noise."""
        from causaltemp_xai.benchmarks.generator import NlinearSCMT
        from causaltemp_xai.metrics.cf_faith import CFfaith

        data = NlinearSCMT(k=4, L=1, T=20, N=10, seed=0, hidden=8).generate(burn_in=20)
        X, graph, mech = data["X"], data["graph"], data["mechanism"]
        X_sel = X[:5]
        rows = _phase07._graph_quality_sweep(
            graph,
            mech,
            X_sel,
            _phase07.build_oracle_interventions(X_sel, mech),
            CFfaith(semantics="noiseless_rollout"),
            1.0,
            method_adj=(graph > 0).astype(int),
            method_auc=1.0,
            method_label="anchor",
            seed=0,
            residual_gt=0.0,
            sigma_x=float(np.std(X_sel)),
        )
        assert all("graph_error_sigma" in r for r in rows)
        true_row = next(r for r in rows if r["label"] == "corrupt_frac=0")
        assert true_row["graph_error_sigma"] == pytest.approx(0.0, abs=1e-9)
        assert all(r["graph_error_sigma"] >= -1e-9 for r in rows)

    def test_scale_free_column_absent_without_a_baseline(self):
        """No ``residual_gt`` means no defensible normalisation baseline, so
        the key must be *omitted* rather than filled with a guess -- a reader
        pooling these payloads must be able to tell "not computed" from
        "computed as zero"."""
        from causaltemp_xai.benchmarks.generator import NlinearSCMT
        from causaltemp_xai.metrics.cf_faith import CFfaith

        data = NlinearSCMT(k=4, L=1, T=20, N=10, seed=0, hidden=8).generate(burn_in=20)
        X, graph, mech = data["X"], data["graph"], data["mechanism"]
        X_sel = X[:3]
        row = _phase07._score_graph_point(
            graph,
            mech,
            X_sel,
            _phase07.build_oracle_interventions(X_sel, mech),
            CFfaith(semantics="noiseless_rollout"),
            1.0,
            "no-baseline",
            (graph > 0).astype(int),
            1.0,
        )
        assert "graph_error_sigma" not in row
        assert "residual_inferred" in row


class TestGraphQualitySweepEnsemble:
    """M4f (2026-08-06): the uncertainty-aware DYNOTEARS ensemble.

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
