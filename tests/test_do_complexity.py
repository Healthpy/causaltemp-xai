"""Adversarial tests for intervention schedules and do-complexity (R6, RISK-18).

The quantity under test is *how the benchmark reads a proposal*, not how good
the proposal is. Contract:

1. an oracle CF of a single ``do()`` has ``D == 1`` and ``delta_trajectory == 0``
   under **both** readings — the positive control must not care;
2. an oracle CF of a genuine 3-slice schedule has ``D == 3``, and the
   single-slice reader posts a large ``delta_trajectory`` on it. That row is
   pinned deliberately: it is the artifact RISK-18 names, reproduced on a CF
   that is *exactly* the world's own trajectory and therefore cannot be wrong;
3. a dense direct edit reaches ``delta_trajectory == 0`` in schedule mode only
   by declaring most of the trajectory intervened, which ``D`` exposes — so the
   pair discriminates where neither number alone does;
4. ``D`` is **monotone** in a tunable amount of causal fidelity (the alpha
   ladder). This is the suite's first graded-sensitivity test; every other
   adversarial test is binary pass/fail, which is what
   ``docs/axis_metrics_report.md``'s "0 open rows" banner cannot speak to;
5. the default (``schedule=False``) is bit-identical to the pre-M2b code path,
   because every committed number depends on it (R7).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
from causaltemp_xai.benchmarks.structural_cf import (
    structural_counterfactual,
    structural_counterfactual_schedule,
)
from causaltemp_xai.metrics.pns import (
    do_complexity,
    do_complexity_stability,
    extract_intervention_schedule,
    pns_direction,
    scm_label,
)
from causaltemp_xai.scm.intervention import is_vacuous_intervention

K, L, T = 4, 1, 30


def _make_scm(k=K, lag=L, T=T, seed=0, noise_scale=0.05):
    """A stable VAR(L) factual trajectory plus its mechanism."""
    rng = np.random.default_rng(seed)
    A_list = [rng.uniform(-0.3, 0.3, (k, k)) for _ in range(lag)]
    x = np.zeros((T, k))
    noise = rng.laplace(0, noise_scale, (T, k))
    for t in range(lag, T):
        for step, A in enumerate(A_list, start=1):
            x[t] += A @ x[t - step]
        x[t] += noise[t]
    return x, LinearMechanism(A_list)


class _ThresholdModel:
    """A classifier that agrees with the SCM label rule."""

    def __init__(self, theta):
        self.theta = theta

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, -1, 0] > self.theta).astype(int)


# ---------------------------------------------------------------------------
# 1. The oracle positive control is invariant to the reading
# ---------------------------------------------------------------------------


class TestSingleSliceOracle:
    def test_single_do_has_do_complexity_one(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual(x, mech, t0=10, node=1, value=3.0, noiseless=False)
        assert do_complexity(x, x_cf, mech) == 1

    def test_schedule_recovers_the_intervention_that_produced_it(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual(x, mech, t0=10, node=1, value=3.0, noiseless=False)
        sched = extract_intervention_schedule(x, x_cf, mech)
        assert len(sched) == 1
        t, nodes, values = sched[0]
        assert t == 10
        assert nodes.tolist() == [1]
        assert values == pytest.approx([3.0])

    def test_replaying_the_extracted_schedule_reproduces_the_cf(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual(x, mech, t0=10, node=1, value=3.0, noiseless=False)
        sched = extract_intervention_schedule(x, x_cf, mech)
        replayed = structural_counterfactual_schedule(x, mech, sched, noiseless=False)
        assert np.allclose(replayed, x_cf, atol=1e-9)

    def test_delta_trajectory_is_zero_under_both_readings(self):
        x, mech = _make_scm()
        theta = 0.0
        model = _ThresholdModel(theta)
        x_cf = structural_counterfactual(x, mech, t0=10, node=0, value=5.0, noiseless=False)
        X, CFs = x[None], x_cf[None]
        for schedule in (False, True):
            out = pns_direction(X, CFs, model, mech, theta, target_class=1, schedule=schedule)
            assert out["delta_trajectory"] == pytest.approx(0.0), f"schedule={schedule}"

    def test_multi_node_single_slice_is_still_one_action(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual(
            x, mech, t0=12, node=[0, 2], value=[2.0, -2.0], noiseless=False
        )
        assert do_complexity(x, x_cf, mech) == 1

    def test_oracle_pearl_batch_mean_is_one(self):
        """The named Pearl oracle is the calibration anchor for the metric."""
        from causaltemp_xai.eval import evaluate_method
        from experiments._common import ORACLE_SHIFT, oracle_intervention_spec

        x, mech = _make_scm()
        X = np.stack([x, x.copy()])
        cfs = np.stack(
            [
                structural_counterfactual(x_i, mech, t0, node, value, noiseless=False)
                for x_i, (t0, node, value) in zip(
                    X, oracle_intervention_spec(X, mech.k, ORACLE_SHIFT)
                )
            ]
        )
        graph = np.zeros((K, K, L))

        class _AllValid:
            def predict(self, Z):
                return np.ones(len(Z), dtype=int)

        out = evaluate_method(_AllValid(), X, cfs, X, graph, mech, target_class=1)
        assert out["do_complexity_mean_pearl_scorable"] == pytest.approx(1.0)
        assert out["n_do_scorable"] == len(X)


# ---------------------------------------------------------------------------
# 2. The RISK-18 artifact, pinned on a CF that is the world's own trajectory
# ---------------------------------------------------------------------------


class TestMultiSliceOracle:
    SCHEDULE: ClassVar[list] = [(8, [1], [3.0]), (14, [2], [-2.5]), (20, [0], [4.0])]

    def test_three_slice_schedule_has_do_complexity_three(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual_schedule(x, mech, self.SCHEDULE, noiseless=False)
        assert do_complexity(x, x_cf, mech) == 3

    def test_extracted_schedule_matches_the_one_imposed(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual_schedule(x, mech, self.SCHEDULE, noiseless=False)
        sched = extract_intervention_schedule(x, x_cf, mech)
        assert [t for t, _, _ in sched] == [8, 14, 20]
        for (t, nodes, values), (t_ref, n_ref, v_ref) in zip(sched, self.SCHEDULE):
            assert t == t_ref
            assert nodes.tolist() == n_ref
            assert values == pytest.approx(v_ref)

    def test_single_slice_reader_misreads_a_genuine_multi_do_cf(self):
        """The artifact. This CF *is* the world's trajectory, so a correct
        reading must score it at zero disagreement — and the single-slice
        reading does not, because it drops the later two interventions."""
        x, mech = _make_scm()
        x_cf = structural_counterfactual_schedule(x, mech, self.SCHEDULE, noiseless=False)

        sched = extract_intervention_schedule(x, x_cf, mech)
        oracle_full = structural_counterfactual_schedule(x, mech, sched, noiseless=False)
        oracle_first = structural_counterfactual(x, mech, 8, [1], [3.0], noiseless=False)

        assert np.allclose(oracle_full, x_cf, atol=1e-9)
        # The single-slice oracle diverges from the proposal it claims to audit.
        assert np.abs(oracle_first - x_cf).max() > 1.0

    def test_schedule_mode_scores_the_multi_do_oracle_at_zero_gap(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual_schedule(x, mech, self.SCHEDULE, noiseless=False)
        theta = 0.0
        model = _ThresholdModel(theta)
        target = scm_label(x_cf, theta)
        out = pns_direction(
            x[None], x_cf[None], model, mech, theta, target_class=target, schedule=True
        )
        assert out["delta_total"] == pytest.approx(0.0)
        assert out["do_complexity_mean"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. A dense direct edit buys delta_trajectory == 0 only at maximal D
# ---------------------------------------------------------------------------


class TestDenseDirectEdit:
    def test_dense_edit_has_near_maximal_do_complexity(self):
        x, mech = _make_scm()
        rng = np.random.default_rng(7)
        x_cf = x + rng.normal(0, 1.0, x.shape)  # edits every cell, no mechanism
        d = do_complexity(x, x_cf, mech)
        assert d >= T - 2, f"expected a near-maximal schedule, got D={d}"

    def test_pair_discriminates_where_delta_trajectory_alone_does_not(self):
        """Both CFs reach delta_trajectory == 0 in schedule mode; only D tells
        the causally coherent one from the direct rewrite."""
        x, mech = _make_scm()
        theta = 0.0
        model = _ThresholdModel(theta)

        causal = structural_counterfactual(x, mech, t0=10, node=0, value=5.0, noiseless=False)
        rng = np.random.default_rng(7)
        dense = x + rng.normal(0, 1.0, x.shape)

        out_causal = pns_direction(
            x[None], causal[None], model, mech, theta, target_class=1, schedule=True
        )
        out_dense = pns_direction(
            x[None], dense[None], model, mech, theta, target_class=1, schedule=True
        )

        assert out_causal["delta_trajectory"] == pytest.approx(0.0)
        assert out_dense["delta_trajectory"] == pytest.approx(0.0)
        assert out_causal["do_complexity_mean"] < out_dense["do_complexity_mean"]

    def test_vacuous_cf_has_do_complexity_zero(self):
        """D == 0 is exactly the RISK-17 vacuity case, and the two detectors
        must agree — D generalises `is_vacuous_intervention`, not duplicates it."""
        x, mech = _make_scm()
        # A noiseless rollout with a zero perturbation: differs from the factual
        # (it deletes the noise) but asserts no action anywhere.
        x_cf = structural_counterfactual(x, mech, t0=10, node=0, value=x[10, 0], noiseless=True)
        assert is_vacuous_intervention(x, x_cf, mech) is True
        assert do_complexity(x, x_cf, mech, semantics="noiseless_rollout") == 0

    def test_noiseless_rollout_is_not_pearl_reproducible(self):
        """The same CF read against a Pearl oracle: deleting the factual noise
        cannot be expressed as a small set of do()s, and D says so."""
        x, mech = _make_scm()
        x_cf = structural_counterfactual(x, mech, t0=10, node=0, value=x[10, 0], noiseless=True)
        assert do_complexity(x, x_cf, mech, semantics="pearl_delta") > 5

    def test_literal_no_op_has_empty_schedule(self):
        x, mech = _make_scm()
        assert extract_intervention_schedule(x, x.copy(), mech) == []
        assert do_complexity(x, x.copy(), mech) == 0

    def test_explicit_batch_denominators_and_semantic_counterexample(self):
        from causaltemp_xai.eval import evaluate_method

        x, mech = _make_scm()
        graph = np.zeros((K, K, L))

        class _AllValid:
            def predict(self, Z):
                return np.ones(len(Z), dtype=int)

        pearl_cf = structural_counterfactual(x, mech, t0=10, node=0, value=3.0, noiseless=False)
        X = np.stack([x, x])
        CFs = np.stack([pearl_cf, x.copy()])
        out = evaluate_method(_AllValid(), X, CFs, X, graph, mech, target_class=1)
        assert out["do_complexity_mean"] == out["do_complexity_mean_all"]
        assert out["n_do_scorable"] == 1
        assert out["frac_no_do_schedule"] == pytest.approx(0.5)
        assert out["do_complexity_mean_all"] == pytest.approx(
            out["do_complexity_mean_pearl_scorable"] * out["n_do_scorable"] / out["n"]
        )

        no_schedules = evaluate_method(_AllValid(), X, X.copy(), X, graph, mech, target_class=1)
        assert no_schedules["n_do_scorable"] == 0
        assert np.isnan(no_schedules["do_complexity_mean_pearl_scorable"])

        # Noiseless vacuity and a non-empty Pearl schedule are compatible; this
        # is the archived full_nl/NoiselessSCMRecourse counterexample in miniature.
        noiseless_cf = structural_counterfactual(
            x, mech, t0=10, node=0, value=float(x[10, 0]), noiseless=True
        )
        counterexample = evaluate_method(
            _AllValid(), x[None], noiseless_cf[None], x[None], graph, mech, target_class=1
        )
        assert counterexample["frac_vacuous"] == 1.0
        assert counterexample["n_do_scorable"] == 1
        assert counterexample["do_complexity_mean_pearl_scorable"] > 0


# ---------------------------------------------------------------------------
# 4. Graded sensitivity: D is monotone in known causal fidelity
# ---------------------------------------------------------------------------


class TestCalibrationLadder:
    ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

    @staticmethod
    def _ladder_cf(x, mech, alpha):
        """Blend the oracle CF (alpha=0) with a direct terminal rewrite (alpha=1).

        Both endpoints move ``x[-1, 0]`` by a comparable amount, so the ladder
        varies *how* the outcome is reached, not how far.
        """
        oracle = structural_counterfactual(x, mech, t0=5, node=0, value=4.0, noiseless=False)
        direct = x.copy()
        direct[5:, 0] = oracle[5:, 0]  # same channel-0 path, no mechanism propagation
        return (1.0 - alpha) * oracle + alpha * direct

    def test_do_complexity_is_monotone_in_alpha(self):
        x, mech = _make_scm()
        ds = [do_complexity(x, self._ladder_cf(x, mech, a), mech) for a in self.ALPHAS]
        assert ds == sorted(ds), f"D not monotone across the ladder: {ds}"
        assert ds[0] < ds[-1], f"ladder has no dynamic range: {ds}"

    def test_endpoints_are_the_expected_extremes(self):
        x, mech = _make_scm()
        assert do_complexity(x, self._ladder_cf(x, mech, 0.0), mech) == 1
        assert do_complexity(x, self._ladder_cf(x, mech, 1.0), mech) > 1


# ---------------------------------------------------------------------------
# 5. Non-regression: the default reading is unchanged
# ---------------------------------------------------------------------------


class TestDefaultUnchanged:
    def test_scalar_form_matches_the_pre_m2b_rollout_exactly(self):
        """`structural_counterfactual` now delegates to the schedule form; it
        must still reproduce the hand-rolled abduction-action-prediction loop
        bit-for-bit, since the oracle control and every committed CF-faith
        number depend on it."""
        from causaltemp_xai.benchmarks.mechanisms import lag_window
        from causaltemp_xai.benchmarks.structural_cf import abduct_noise

        x, mech = _make_scm()
        t0, node, value = 10, 1, 3.0

        eps = abduct_noise(x, mech)
        expected = x.copy()
        expected[t0, node] = value
        for t in range(t0 + 1, T):
            expected[t] = mech.forward_numpy(lag_window(expected, t, L, K)) + eps[t]

        got = structural_counterfactual(x, mech, t0, node, value, noiseless=False)
        assert np.array_equal(got, expected)

    def test_noiseless_variant_unchanged(self):
        x, mech = _make_scm()
        got = structural_counterfactual(x, mech, 10, 1, 3.0, noiseless=True)
        sched = structural_counterfactual_schedule(x, mech, [(10, 1, 3.0)], noiseless=True)
        assert np.array_equal(got, sched)

    def test_pns_default_is_the_single_slice_reading(self):
        """schedule=False must not silently pick up the new behaviour."""
        x, mech = _make_scm()
        theta = 0.0
        model = _ThresholdModel(theta)
        x_cf = structural_counterfactual_schedule(
            x, mech, [(8, [1], [3.0]), (14, [2], [-2.5])], noiseless=False
        )
        default = pns_direction(x[None], x_cf[None], model, mech, theta, target_class=1)
        assert default["schedule_mode"] is False
        # D is reported even when it is not acted on, so a single-slice run
        # exposes how much of the proposal it declined to model.
        assert default["do_complexity_mean"] == pytest.approx(2.0)


class TestToleranceRobustness:
    """The group separation must not be a property of `INTERVENTION_TOL`.

    Measured on real CFs (`docs/cf_faith_methodology.md` §10), some methods'
    exact `D` *is* tolerance-sensitive — CftsCels swings 1.0 -> 13.2 across three
    orders of magnitude. What the contribution rests on is the separation
    between a single-`do()` proposal and a dense rewrite, and that is pinned
    here so a future tolerance change cannot silently erase it.
    """

    def test_single_do_stays_at_one_across_tolerances(self):
        x, mech = _make_scm()
        x_cf = structural_counterfactual(x, mech, t0=10, node=1, value=3.0, noiseless=False)
        for tol in (1e-2, 1e-3, 1e-4):
            assert do_complexity(x, x_cf, mech, tol=tol) == 1, tol

    def test_dense_edit_stays_near_maximal_across_tolerances(self):
        x, mech = _make_scm()
        rng = np.random.default_rng(7)
        x_cf = x + rng.normal(0, 1.0, x.shape)
        for tol in (1e-2, 1e-3, 1e-4):
            assert do_complexity(x, x_cf, mech, tol=tol) >= T - 2, tol

    def test_separation_survives_every_tolerance(self):
        x, mech = _make_scm()
        causal = structural_counterfactual(x, mech, t0=10, node=1, value=3.0, noiseless=False)
        rng = np.random.default_rng(7)
        dense = x + rng.normal(0, 1.0, x.shape)
        for tol in (1e-2, 1e-3, 1e-4):
            d_causal = do_complexity(x, causal, mech, tol=tol)
            d_dense = do_complexity(x, dense, mech, tol=tol)
            assert d_dense > 10 * d_causal, (tol, d_causal, d_dense)


class TestScheduleValidation:
    def test_duplicate_timestep_raises(self):
        x, mech = _make_scm()
        with pytest.raises(ValueError, match="two entries for timestep"):
            structural_counterfactual_schedule(x, mech, [(5, 0, 1.0), (5, 1, 2.0)])

    def test_empty_schedule_raises(self):
        x, mech = _make_scm()
        with pytest.raises(ValueError, match="empty"):
            structural_counterfactual_schedule(x, mech, [])

    def test_out_of_range_timestep_raises(self):
        x, mech = _make_scm()
        with pytest.raises(ValueError, match="out of range"):
            structural_counterfactual_schedule(x, mech, [(T + 3, 0, 1.0)])

    def test_shape_mismatch_raises(self):
        x, mech = _make_scm()
        with pytest.raises(ValueError, match="same shape"):
            structural_counterfactual_schedule(x, mech, [(5, [0, 1], [1.0])])

    def test_unknown_semantics_raises(self):
        x, mech = _make_scm()
        with pytest.raises(ValueError, match="unknown semantics"):
            extract_intervention_schedule(x, x.copy() + 1.0, mech, semantics="bogus")

    def test_schedule_order_does_not_matter(self):
        x, mech = _make_scm()
        a = structural_counterfactual_schedule(x, mech, [(8, 1, 3.0), (14, 2, -2.5)])
        b = structural_counterfactual_schedule(x, mech, [(14, 2, -2.5), (8, 1, 3.0)])
        assert np.array_equal(a, b)


class TestThresholdStabilityDiagnostic:
    """`D` is threshold-stable iff the method's actions are bimodal (RISK-20).

    The stability ratio exists because rescaling the threshold does *not* fix a
    method whose edit mass sits inside the threshold band — measured, not
    assumed: switching to a per-channel relative tolerance left CftsCels's swing
    at 12.7x against 13.2x absolute.
    """

    def test_bimodal_cf_is_threshold_independent(self):
        x, mech = _make_scm()
        # One large, unambiguous do(): far above any threshold in the sweep.
        x_cf = structural_counterfactual(x, mech, t0=10, node=1, value=3.0, noiseless=False)
        assert do_complexity_stability(x, x_cf, mech) == pytest.approx(1.0)

    def test_edits_inside_the_threshold_band_are_flagged(self):
        """A CF whose actions sit *at* the tolerance must not look stable."""
        x, mech = _make_scm()
        x_cf = x.copy()
        rng = np.random.default_rng(3)
        # Deviations spread across the swept decades (1e-4 .. 1e-2).
        for t in range(6, T):
            x_cf[t, 0] += 10 ** rng.uniform(-4, -2)
        assert do_complexity_stability(x, x_cf, mech) > 2.0

    def test_stability_is_reported_alongside_D(self):
        from causaltemp_xai.eval import evaluate_method

        x, mech = _make_scm()
        X = np.stack([x, x])
        CFs = np.stack(
            [structural_counterfactual(x, mech, 10, 1, 3.0, noiseless=False) for _ in range(2)]
        )

        class _Clf:
            def predict(self, Z):
                return np.ones(len(np.atleast_3d(Z)), dtype=int)

        out = evaluate_method(_Clf(), X, CFs, X, np.zeros((K, K, 1)), mech, target_class=1)
        assert "do_complexity_stability" in out
        assert out["do_complexity_stability"] == pytest.approx(1.0)
