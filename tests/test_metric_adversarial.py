"""Adversarial metric tests (M1 metric-integrity milestone, objective O1).

Contract: every Axis-C metric and CF-faith must (a) **flag** a constructed
violating CF and (b) **pass** the oracle/compliant CF. A metric that cannot
tell a violator from an oracle measures nothing. (TRSI is the one exception —
it is a mechanism-free *descriptor*, so it is only held to ordering edits by
temporal coherence; certifying causal violations is CF-faith's job.)

Also contains the M1 regression tests:

* **CELS-style false flag** — a CF that reconstructs the full trajectory with
  per-element noise *below* the ``derive_intervention_t`` tolerance must NOT
  be flagged retroactive by CF-faith's retro gate (the pre-M1 summed-1e-4 gate
  falsely produced ``soft=0.0`` on every CELS instance).
* **MCC_coverage discrimination** — the redesigned chance-normalized mass
  formulation must be scale-invariant and must NOT sit at a ceiling for dense
  maps (the pre-M1 absolute threshold ``1e-3`` was trivially cleared).
* **shift_vr guard** — the retention ratio is only reported when
  ``validity_base >= MIN_VALIDITY_BASE_FOR_RATIO``; the
  ``(validity_base, validity_shift)`` pair is always reported.
* **Joint faithfulness-validity criterion** — a tiny-edit CF that is
  hard-faithful but does not flip the classifier earns no joint credit
  (anti-gameability).
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
from causaltemp_xai.eval import MIN_VALIDITY_BASE_FOR_RATIO, evaluate_method, shift_vr
from causaltemp_xai.metrics.axis_a import icc, mcc_concept
from causaltemp_xai.metrics.axis_c import (
    ood_plausibility,
    proximity,
    sparsity,
    trsi,
    validity,
)
from causaltemp_xai.metrics.cf_faith import CFfaith
from causaltemp_xai.scm.intervention import INTERVENTION_TOL, derive_intervention_t


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_scm(k=4, L=1, T=30, seed=0, noise_scale=0.05):
    """(x_original, graph, mechanism) from a small stable VAR(L)."""
    rng = np.random.default_rng(seed)
    A_list = [rng.uniform(-0.3, 0.3, (k, k)) for _ in range(L)]
    graph = np.stack([(A != 0).astype(float) for A in A_list], axis=-1)
    x = np.zeros((T, k))
    noise = rng.laplace(0, noise_scale, (T, k))
    for t in range(L, T):
        for lag, A in enumerate(A_list, start=1):
            x[t] += A @ x[t - lag]
        x[t] += noise[t]
    return x, graph, LinearMechanism(A_list)


def _noiseless_cf(x, mechanism, t0, pert):
    """Oracle-compliant CF: prefix copied, intervened at t0, noiseless rollout."""
    T, k = x.shape
    cf = x.copy()
    cf[t0] = x[t0] + pert
    for t in range(t0 + 1, T):
        nxt = np.zeros(k)
        for l, A in enumerate(mechanism.A_list):
            lag_t = t - l - 1
            if lag_t >= 0:
                nxt += A @ cf[lag_t]
        cf[t] = nxt
    return cf


def _pearl_cf(x, mechanism, t0, pert):
    """Oracle-compliant Pearl CF: abduct factual noise, intervene, re-inject."""
    T, k = x.shape
    # Abduction (exact under additive noise).
    e = np.zeros((T, k))
    for t in range(T):
        pred = np.zeros(k)
        for l, A in enumerate(mechanism.A_list):
            lag_t = t - l - 1
            if lag_t >= 0:
                pred += A @ x[lag_t]
        e[t] = x[t] - pred
    cf = x.copy()
    cf[t0] = x[t0] + pert
    for t in range(t0 + 1, T):
        nxt = e[t].copy()
        for l, A in enumerate(mechanism.A_list):
            lag_t = t - l - 1
            if lag_t >= 0:
                nxt += A @ cf[lag_t]
        cf[t] = nxt
    return cf


class _SignModel:
    """Deterministic mock classifier: class 1 iff x[0, 0] > 0."""

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0, 0] > 0).astype(int)


# ---------------------------------------------------------------------------
# Axis C: validity
# ---------------------------------------------------------------------------


class TestValidityAdversarial:
    def test_flags_non_flipping_cf(self):
        cfs = -np.ones((5, 10, 3))  # x[0,0] < 0 -> class 0, not target
        assert validity(cfs, _SignModel(), target_class=1) == 0.0

    def test_passes_flipping_cf(self):
        cfs = np.ones((5, 10, 3))
        assert validity(cfs, _SignModel(), target_class=1) == 1.0


# ---------------------------------------------------------------------------
# Axis C: proximity (L1 / L2)
# ---------------------------------------------------------------------------


class TestProximityAdversarial:
    def test_flags_distant_cf(self):
        x = np.zeros((10, 3))
        cf_far = x + 10.0
        cf_near = x.copy()
        cf_near[5, 0] += 0.1
        for norm in ("l1", "l2"):
            assert proximity(x, cf_far, norm=norm) > proximity(x, cf_near, norm=norm)

    def test_passes_identical_cf(self):
        x = np.arange(30, dtype=float).reshape(10, 3)
        assert proximity(x, x, norm="l1") == 0.0
        assert proximity(x, x, norm="l2") == 0.0


# ---------------------------------------------------------------------------
# Axis C: sparsity
# ---------------------------------------------------------------------------


class TestSparsityAdversarial:
    def test_flags_dense_edit(self):
        x = np.zeros((10, 3))
        assert sparsity(x, x + 1.0) == 0.0

    def test_passes_single_entry_edit(self):
        x = np.zeros((10, 3))
        cf = x.copy()
        cf[4, 1] = 2.0
        assert sparsity(x, cf) == pytest.approx((30 - 1) / 30)


# ---------------------------------------------------------------------------
# Axis C: OOD plausibility (IsolationForest)
# ---------------------------------------------------------------------------


class TestOODAdversarial:
    def _train(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=(60, 10, 3))

    def test_flags_far_out_of_distribution_cf(self):
        X_train = self._train()
        cf_out = 50.0 * np.ones((10, 3))
        cf_in = X_train[0]
        score_out = ood_plausibility(X_train, cf_out)
        score_in = ood_plausibility(X_train, cf_in)
        assert score_out < score_in
        assert score_out < 0  # IsolationForest flags it as anomalous outright


# ---------------------------------------------------------------------------
# Axis C: TRSI (mechanism-free temporal smoothness of the edit)
#
# TRSI is a descriptor, not a faithfulness criterion, so the contract here is
# only that it ORDERS edits by temporal coherence — it is not asked to flag a
# causal violator (CF-faith's job) and must not be tested as though it did.
# ---------------------------------------------------------------------------


class TestTRSIAdversarial:
    def test_flags_jumpy_edit(self):
        x = np.zeros((20, 3))
        cf_jumpy = x.copy()
        cf_jumpy[::2] += 1.0  # alternating delta: 1,0,1,0,...
        cf_smooth = x + 1.0  # constant delta
        assert trsi(cf_jumpy, x) > trsi(cf_smooth, x)

    def test_passes_constant_shift(self):
        x = np.zeros((20, 3))
        assert trsi(x + 1.0, x) == 0.0  # constant delta has zero 2nd difference


# ---------------------------------------------------------------------------
# CF-faith: violator flagged, oracle passed (both semantics)
# ---------------------------------------------------------------------------


class TestCFfaithAdversarial:
    def test_flags_retroactive_violator_both_semantics(self):
        x, graph, mech = _make_scm(seed=1)
        t0 = 12
        cf = x.copy()
        cf[3, 1] += 5.0  # blatant retroactive edit
        for sem in CFfaith.SEMANTICS:
            r = CFfaith(semantics=sem).score(x, cf, t0, graph, mech)
            assert r == {"hard": 0.0, "soft": 0.0}, f"{sem} must flag retro edit"

    def test_flags_scm_ignorant_cf(self):
        """A CF that edits arbitrarily from t0 on (no SCM consultation) fails
        the forward check under both semantics."""
        x, graph, mech = _make_scm(seed=2)
        t0 = 10
        rng = np.random.default_rng(0)
        cf = x.copy()
        cf[t0:] += rng.normal(scale=0.5, size=cf[t0:].shape)
        for sem in CFfaith.SEMANTICS:
            r = CFfaith(semantics=sem).score(x, cf, t0, graph, mech)
            assert r["hard"] == 0.0, f"{sem} must flag an SCM-ignorant CF"

    def test_passes_noiseless_rollout_oracle(self):
        x, graph, mech = _make_scm(seed=3)
        t0 = 10
        rng = np.random.default_rng(1)
        cf = _noiseless_cf(x, mech, t0, rng.uniform(-0.4, 0.4, x.shape[1]))
        r = CFfaith(semantics="noiseless_rollout").score(x, cf, t0, graph, mech)
        assert r["hard"] == 1.0
        assert r["soft"] > 0.99

    def test_passes_pearl_oracle(self):
        x, graph, mech = _make_scm(seed=4)
        t0 = 10
        rng = np.random.default_rng(2)
        cf = _pearl_cf(x, mech, t0, rng.uniform(-0.4, 0.4, x.shape[1]))
        r = CFfaith(semantics="pearl_delta").score(x, cf, t0, graph, mech)
        assert r["hard"] == 1.0
        assert r["soft"] > 0.99


# ---------------------------------------------------------------------------
# Regression: the CELS-style false flag (M1 fix #1)
# ---------------------------------------------------------------------------


class TestCelsFalseFlagRegression:
    """A method that reconstructs the *entire* trajectory carries float-scale
    per-element reconstruction noise before t0. Pre-M1, the retroactive gate
    summed that noise over the T×k pre-window against 1e-4 and flagged every
    instance (IVR=1.0, soft=0.0 exactly). Post-M1 the gate uses the same
    per-element predicate as derive_intervention_t and must NOT fire."""

    def _cels_style_cf(self, seed=7, t0=10, noise_amp=5e-4):
        x, graph, mech = _make_scm(k=5, L=1, T=30, seed=seed)
        rng = np.random.default_rng(seed + 100)
        cf = _noiseless_cf(x, mech, t0, np.full(x.shape[1], 0.5))
        # Full-trajectory reconstruction noise: every pre-t0 element nonzero
        # but individually below INTERVENTION_TOL. Summed over the pre-window
        # (10 steps x 5 channels) it is ~10-100x the old 1e-4 gate.
        pre_noise = rng.uniform(-noise_amp, noise_amp, size=(t0, x.shape[1]))
        assert np.abs(pre_noise).max() < INTERVENTION_TOL
        assert np.abs(pre_noise).sum() > 10 * 1e-4  # would trip the OLD gate
        cf[:t0] += pre_noise
        return x, graph, mech, cf, t0

    def test_derived_t0_matches_true_intervention(self):
        x, graph, mech, cf, t0 = self._cels_style_cf()
        assert derive_intervention_t(x, cf) == t0

    def test_cf_faith_rollout_not_false_flagged(self):
        """The reconstruction-noise CF is still a valid noiseless rollout from
        t0 (L=1: the forward simulation never reads the pre-t0 rows), so it
        must score hard=1 — pre-M1 it scored hard=0, soft=0.0 exactly."""
        x, graph, mech, cf, t0 = self._cels_style_cf()
        r = CFfaith(semantics="noiseless_rollout").score(x, cf, t0, graph, mech)
        assert r["hard"] == 1.0
        assert r["soft"] > 0.99

    def test_cf_faith_pearl_soft_not_forced_zero(self):
        """Pearl semantics may legitimately reject a rollout-built CF
        (hard=0), but the soft score must be the forward residual's value —
        not the retroactive gate's forced exact 0.0."""
        x, graph, mech, cf, t0 = self._cels_style_cf()
        p = CFfaith(semantics="pearl_delta").score(x, cf, t0, graph, mech)
        assert p["soft"] > 0.0

    def test_pearl_built_cels_style_cf_passes_pearl(self):
        """Mirror case: Pearl-oracle CF + sub-tolerance reconstruction noise
        must keep pearl hard=1."""
        x, graph, mech = _make_scm(k=5, L=1, T=30, seed=11)
        t0 = 10
        rng = np.random.default_rng(42)
        cf = _pearl_cf(x, mech, t0, np.full(x.shape[1], 0.5))
        cf[:t0] += rng.uniform(-5e-4, 5e-4, size=(t0, x.shape[1]))
        p = CFfaith(semantics="pearl_delta").score(x, cf, t0, graph, mech)
        assert p["hard"] == 1.0

    def test_genuine_retroactive_edit_still_flagged(self):
        """The fix must not open the gate to real violations: one pre-t0
        element above the tolerance is still a retroactive edit."""
        x, graph, mech, cf, t0 = self._cels_style_cf()
        cf[4, 2] += 0.01  # a genuine (small but super-tolerance) retro edit
        for sem in CFfaith.SEMANTICS:
            r = CFfaith(semantics=sem).score(x, cf, t0, graph, mech)
            assert r == {"hard": 0.0, "soft": 0.0}


# ---------------------------------------------------------------------------
# Axis A: ICC + MCC_coverage (redesigned) — adversarial + regression
# ---------------------------------------------------------------------------


class TestICCAdversarial:
    def test_flags_wrong_channel_attribution(self):
        att = np.zeros((20, 4))
        att[:, 2] = 1.0
        assert icc(att, int_channel=0) == 0.0

    def test_passes_correct_channel_attribution(self):
        att = np.zeros((20, 4))
        att[:, 0] = 1.0
        assert icc(att, int_channel=0) == 1.0


class TestMCCCoverageFix:
    K = 4
    PARENTS = [0, 1]

    def test_flags_parent_avoiding_attribution(self):
        att = np.zeros((20, self.K))
        att[:, 2] = att[:, 3] = 1.0  # all mass off the parents
        assert mcc_concept(att, self.PARENTS) == 0.0

    def test_passes_parent_concentrated_attribution(self):
        att = np.zeros((20, self.K))
        att[:, 0] = att[:, 1] = 1.0  # all mass on the parents
        # Maximum score = k / |Pa| = 2.0
        assert mcc_concept(att, self.PARENTS) == pytest.approx(self.K / len(self.PARENTS))

    def test_uniform_map_scores_chance_level(self):
        att = np.ones((20, self.K))
        assert mcc_concept(att, self.PARENTS) == pytest.approx(1.0)

    def test_scale_invariance(self):
        """Pre-M1 the absolute 1e-3 threshold made the score depend on the
        attribution's output scale; the mass formulation must not."""
        rng = np.random.default_rng(0)
        att = rng.normal(size=(20, self.K))
        s1 = mcc_concept(att, self.PARENTS)
        s2 = mcc_concept(1e-6 * att, self.PARENTS)
        s3 = mcc_concept(1e6 * att, self.PARENTS)
        assert s1 == pytest.approx(s2) == pytest.approx(s3)

    def test_no_ceiling_for_dense_maps(self):
        """Regression for the ceiling effect: dense (IG-like) maps must not
        pin the metric at its maximum, and different maps must get different
        scores (discriminative power restored)."""
        rng = np.random.default_rng(1)
        scores = [
            mcc_concept(np.abs(rng.normal(size=(20, self.K))), self.PARENTS)
            for _ in range(5)
        ]
        ceiling = self.K / len(self.PARENTS)
        assert all(s < ceiling for s in scores)
        assert len({round(s, 12) for s in scores}) > 1  # not all identical

    def test_empty_parents_is_nan(self):
        assert np.isnan(mcc_concept(np.ones((20, self.K)), []))

    def test_zero_mass_map_is_nan(self):
        assert np.isnan(mcc_concept(np.zeros((20, self.K)), self.PARENTS))


# ---------------------------------------------------------------------------
# Axis D: shift_vr guard (M1 fix #3)
# ---------------------------------------------------------------------------


class _FractionMethod:
    """Mock CF method: 1st call (base) / 2nd call (shift) return batches where
    a fixed fraction of instances is marked target-class for _SignModel."""

    def __init__(self, frac_base, frac_shift):
        self._fracs = [frac_base, frac_shift]
        self._calls = 0

    def generate_batch(self, X, model):
        frac = self._fracs[self._calls]
        self._calls += 1
        X = np.asarray(X, dtype=float).copy()
        n_pos = int(round(frac * X.shape[0]))
        X[:, 0, 0] = -1.0
        X[:n_pos, 0, 0] = 1.0
        return X


class TestShiftVRGuard:
    def _run(self, methods):
        X = np.zeros((10, 8, 3))
        return shift_vr(_SignModel(), methods, X, X, graph=None, mechanism=None,
                        target_class=1)

    def test_ratio_suppressed_below_floor_pair_still_reported(self):
        res = self._run({"unstable": _FractionMethod(0.2, 0.8)})["unstable"]
        assert res["validity_base"] == pytest.approx(0.2)
        assert res["validity_shift"] == pytest.approx(0.8)
        assert res["shift_vr"] is None  # 4.0x artifact must NOT be reported

    def test_zero_base_gives_none_not_nan(self):
        res = self._run({"dead": _FractionMethod(0.0, 0.5)})["dead"]
        assert res["shift_vr"] is None

    def test_ratio_reported_at_or_above_floor(self):
        res = self._run({"stable": _FractionMethod(0.5, 0.4)})["stable"]
        assert res["shift_vr"] == pytest.approx(0.8)

    def test_floor_boundary_inclusive(self):
        res = self._run({"edge": _FractionMethod(MIN_VALIDITY_BASE_FOR_RATIO, 0.3)})["edge"]
        assert res["shift_vr"] == pytest.approx(1.0)

    def test_sample_sizes_reported(self):
        res = self._run({"m": _FractionMethod(0.5, 0.5)})["m"]
        assert res["n_base"] == 10 and res["n_shift"] == 10


# ---------------------------------------------------------------------------
# Axis D: shift_vr base-CF reuse (2026-07-15)
#
# Regenerating the base CFs was both wasted work (Phase 03 already generated
# and persisted exactly these arrays) and *wrong* for stochastic methods: the
# throwaway regeneration is a different CF set than the persisted one that
# Phase 04 scores, so validity_base described counterfactuals nothing else in
# the pipeline ever saw. `cf_base` pins the denominator to the real arrays.
# ---------------------------------------------------------------------------


class _StochasticMethod:
    """Mock CF method whose output differs on every call (like CftsCounts)."""

    def __init__(self, fracs):
        self._fracs = list(fracs)
        self.calls = 0

    def generate_batch(self, X, model):
        frac = self._fracs[min(self.calls, len(self._fracs) - 1)]
        self.calls += 1
        X = np.asarray(X, dtype=float).copy()
        X[:, 0, 0] = -1.0
        X[: int(round(frac * X.shape[0])), 0, 0] = 1.0
        return X


def _batch_with_validity(frac, n=10, T=8, k=3):
    """A (n, T, k) batch that _SignModel scores at exactly ``frac`` validity."""
    X = np.zeros((n, T, k))
    X[:, 0, 0] = -1.0
    X[: int(round(frac * n)), 0, 0] = 1.0
    return X


class TestShiftVRBaseReuse:
    X = np.zeros((10, 8, 3))

    def test_reuse_skips_base_generation(self):
        m = _StochasticMethod([0.4, 0.5])
        shift_vr(_SignModel(), {"m": m}, self.X, self.X, None, None, 1,
                 cf_base={"m": _batch_with_validity(0.6)})
        assert m.calls == 1, "base CFs must not be regenerated when supplied"

    def test_validity_base_comes_from_supplied_array(self):
        """The correctness fix: for a stochastic method the reported
        validity_base must describe the *persisted* CFs (0.6), not whatever a
        fresh regeneration happens to produce. With reuse the only generation
        left is the shift call, so the mock yields 0.5 there."""
        m = _StochasticMethod([0.5])
        res = shift_vr(_SignModel(), {"m": m}, self.X, self.X, None, None, 1,
                       cf_base={"m": _batch_with_validity(0.6)})["m"]
        assert res["validity_base"] == pytest.approx(0.6)
        assert res["validity_shift"] == pytest.approx(0.5)  # numerator still fresh
        assert res["shift_vr"] == pytest.approx(0.5 / 0.6)

    def test_missing_method_falls_back_to_generating(self):
        m = _StochasticMethod([0.5, 0.4])
        res = shift_vr(_SignModel(), {"m": m}, self.X, self.X, None, None, 1,
                       cf_base={"other": _batch_with_validity(0.6)})["m"]
        assert m.calls == 2  # base + shift, as before
        assert res["validity_base"] == pytest.approx(0.5)

    def test_omitting_cf_base_preserves_old_behaviour(self):
        m = _StochasticMethod([0.5, 0.4])
        res = shift_vr(_SignModel(), {"m": m}, self.X, self.X, None, None, 1)["m"]
        assert m.calls == 2
        assert res["validity_base"] == pytest.approx(0.5)
        assert res["shift_vr"] == pytest.approx(0.8)

    def test_reuse_is_faithful_for_a_deterministic_method(self):
        """With a deterministic method, reuse must change nothing at all."""
        base_cfs = _FractionMethod(0.5, 0.5).generate_batch(self.X, None)
        without = shift_vr(_SignModel(), {"m": _FractionMethod(0.5, 0.4)},
                           self.X, self.X, None, None, 1)["m"]
        with_reuse = shift_vr(_SignModel(), {"m": _StochasticMethod([0.4])},
                              self.X, self.X, None, None, 1,
                              cf_base={"m": base_cfs})["m"]
        assert with_reuse["validity_base"] == pytest.approx(without["validity_base"])
        assert with_reuse["shift_vr"] == pytest.approx(without["shift_vr"])


# ---------------------------------------------------------------------------
# Joint faithfulness-validity criterion (M1 fix #4, anti-gameability)
# ---------------------------------------------------------------------------


T0_JOINT = 7


class _StepValueModel:
    """Deterministic mock classifier keyed to the intervention step: class 1
    iff x[T0_JOINT, 0] > 0.5. A tiny (sub-0.5) intervention cannot flip it; a
    large one always does — both CFs stay *pure noiseless rollouts*."""

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, T0_JOINT, 0] > 0.5).astype(int)


def _make_batch_scm(k=3, L=1, T=15, n=6, seed=13, noise_scale=0.01):
    """One shared mechanism, ``n`` factual trajectories (as evaluate_method
    assumes: a single SCM per benchmark)."""
    rng = np.random.default_rng(seed)
    A_list = [rng.uniform(-0.3, 0.3, (k, k)) for _ in range(L)]
    graph = np.stack([(A != 0).astype(float) for A in A_list], axis=-1)
    mech = LinearMechanism(A_list)
    X = np.zeros((n, T, k))
    noise = rng.laplace(0, noise_scale, (n, T, k))
    for t in range(L, T):
        for lag, A in enumerate(A_list, start=1):
            X[:, t] += X[:, t - lag] @ A.T
        X[:, t] += noise[:, t]
    return X, graph, mech


class TestJointFaithValidCriterion:
    def _batches(self):
        X, graph, mech = _make_batch_scm()
        # Sanity: trajectories are small, so the 0.5 decision threshold at
        # T0_JOINT separates tiny from large interventions cleanly.
        assert np.abs(X[:, T0_JOINT, 0]).max() < 0.4
        # Tiny edit: hard-faithful rollout, above INTERVENTION_TOL (so it IS
        # the derived intervention) but far too small to flip the model ->
        # the gameability case.
        tiny = np.stack([
            _noiseless_cf(x, mech, T0_JOINT, np.full(3, 2e-3)) for x in X
        ])
        # Large edit: hard-faithful rollout that also flips the model.
        big = np.stack([
            _noiseless_cf(x, mech, T0_JOINT, np.full(3, 3.0)) for x in X
        ])
        return X, graph, mech, tiny, big

    def test_tiny_edit_gets_no_joint_credit(self):
        """Hard-faithful but non-flipping CFs: faithfulness alone is at 1.0,
        joint credit is zero — the loophole is closed."""
        X, graph, mech, tiny, _ = self._batches()
        result = evaluate_method(_StepValueModel(), X, tiny, X, graph, mech,
                                 target_class=1)
        assert result["cf_faith_rollout_hard"] == 1.0  # gameably 'faithful'
        assert result["validity"] == 0.0               # ...but flips nothing
        assert result["cf_faith_rollout_hard_valid"] == 0.0  # no joint credit

    def test_faithful_and_valid_cf_gets_full_joint_credit(self):
        X, graph, mech, _, big = self._batches()
        result = evaluate_method(_StepValueModel(), X, big, X, graph, mech,
                                 target_class=1)
        assert result["validity"] == 1.0
        assert result["cf_faith_rollout_hard"] == 1.0
        assert result["cf_faith_rollout_hard_valid"] == 1.0

    def test_joint_credit_requires_both(self):
        """Mixed batch: joint score == mean(hard_i * valid_i), strictly below
        the faithfulness marginal when the tiny-edit half flips nothing."""
        X, graph, mech, tiny, big = self._batches()
        X2 = np.concatenate([X, X])
        CFs = np.concatenate([tiny, big])
        result = evaluate_method(_StepValueModel(), X2, CFs, X, graph, mech,
                                 target_class=1)
        preds = _StepValueModel().predict(CFs)
        rollout = CFfaith(semantics="noiseless_rollout")
        manual = []
        for i in range(len(X2)):
            t = derive_intervention_t(X2[i], CFs[i])
            hard = rollout.score(X2[i], CFs[i], t, graph, mech)["hard"]
            manual.append(hard * float(preds[i] == 1))
        assert result["cf_faith_rollout_hard_valid"] == pytest.approx(
            float(np.mean(manual))
        )
        # The tiny-edit half contributes faithfulness but no joint credit.
        assert result["cf_faith_rollout_hard_valid"] < result["cf_faith_rollout_hard"]
