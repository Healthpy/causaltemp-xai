"""Tests for CFfaith — causal-faithfulness scorer.

Verifies:
- Perfectly SCM-compliant CF → hard=1.0, soft close to 1.0.
- CF with retroactive change (before intervention_t) → hard=0.0, soft=0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmark.mechanisms import LinearMechanism
from causaltemp_xai.metrics.cf_faith import CFfaith


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_scm(k: int = 3, L: int = 1, T: int = 20, seed: int = 0):
    """Return (x_original, graph, mechanism) for a simple VAR(L) system."""
    rng = np.random.default_rng(seed)
    # Random stable coefficient matrices
    A_list = []
    for _ in range(L):
        A = rng.uniform(-0.3, 0.3, (k, k))
        A_list.append(A)

    graph = np.stack([(A != 0).astype(float) for A in A_list], axis=-1)  # (k, k, L)

    # Simulate T time steps
    x = np.zeros((T, k))
    noise = rng.laplace(0, 0.01, (T, k))
    for t in range(L, T):
        for lag, A in enumerate(A_list, start=1):
            x[t] += A @ x[t - lag]
        x[t] += noise[t]

    return x, graph, LinearMechanism(A_list)


# ---------------------------------------------------------------------------
# SCM-compliant CF
# ---------------------------------------------------------------------------


class TestCFFaithCompliant:
    def test_hard_score_is_one(self):
        """CF exactly obeying the SCM forward from intervention_t → hard=1."""
        k, L, T = 4, 1, 30
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=1)
        intervention_t = 10
        rng = np.random.default_rng(42)

        # Build a CF: identical up to intervention_t, then perturb at intervention_t
        # and propagate forward strictly via the SCM mechanism.
        x_cf = x_orig.copy()
        x_cf[intervention_t] += rng.uniform(-0.5, 0.5, k)  # intervention

        # Propagate exactly via mechanism (no noise)
        for t in range(intervention_t + 1, T):
            x_cf[t] = np.zeros(k)
            for lag, A in enumerate(mechanism.A_list, start=1):
                if t - lag >= 0:
                    x_cf[t] += A @ x_cf[t - lag]

        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanism)
        assert result["hard"] == 1.0, f"Expected hard=1.0, got {result['hard']}"

    def test_soft_score_near_one(self):
        """SCM-compliant CF → soft score should be close to 1.0."""
        k, L, T = 3, 1, 25
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=2)
        intervention_t = 8
        rng = np.random.default_rng(7)

        x_cf = x_orig.copy()
        x_cf[intervention_t] += rng.uniform(-0.3, 0.3, k)
        for t in range(intervention_t + 1, T):
            x_cf[t] = np.zeros(k)
            for lag, A in enumerate(mechanism.A_list, start=1):
                if t - lag >= 0:
                    x_cf[t] += A @ x_cf[t - lag]

        scorer = CFfaith(tol=1e-3, scale=1.0)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanism)
        assert result["soft"] > 0.9, f"Expected soft>0.9, got {result['soft']}"


# ---------------------------------------------------------------------------
# Retroactive CF
# ---------------------------------------------------------------------------


class TestCFFaithRetroactive:
    def test_hard_score_is_zero(self):
        """CF with a retroactive change (before intervention_t) → hard=0.0."""
        k, L, T = 4, 1, 30
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=3)
        intervention_t = 15

        x_cf = x_orig.copy()
        # Retroactive: modify a time step BEFORE intervention_t
        x_cf[5] += 999.0  # blatant change before intervention

        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanism)
        assert result["hard"] == 0.0, f"Expected hard=0.0, got {result['hard']}"

    def test_soft_score_is_zero_on_retroactive(self):
        """Retroactive CF → soft=0.0 (early-return path)."""
        k, L, T = 3, 1, 20
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=4)
        intervention_t = 10

        x_cf = x_orig.copy()
        x_cf[3, 0] += 5.0  # retroactive change

        scorer = CFfaith(tol=1e-4)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanism)
        assert result["soft"] == 0.0, f"Expected soft=0.0, got {result['soft']}"

    def test_identical_cf_hard_is_zero(self):
        """Identical CF (null intervention) → hard=0 under noiseless-rollout semantics.

        CF-faith (noiseless-rollout definition) asks whether ``x_cf[t0:]`` *is* the
        deterministic, noiseless VAR continuation of itself. The factual trajectory
        carries innovation noise (scale ≈ 0.1 in the real generator), so a CF equal
        to the factual deviates from its own noiseless rollout by ~the noise and is
        correctly judged unfaithful. Only CFs that are themselves noiseless SCM
        rollouts from the intervention point (e.g. CARLA-causal) score hard=1 — that
        is the property the benchmark uses to separate causal from arbitrary CFs.
        A do-nothing CF is degenerate (it flips no label) and is never produced by a
        real method; see index "Decisions" (2026-05-30, CF-faith semantics).
        """
        k, L, T = 3, 1, 20
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=5)
        intervention_t = 5
        x_cf = x_orig.copy()

        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanism)
        assert result["hard"] == 0.0


# ---------------------------------------------------------------------------
# Pearl delta-recursion semantics (the second, on-manifold metric)
# ---------------------------------------------------------------------------


def _abduct(x_orig, mechanism):
    """Recover the innovation sequence e[t] = x[t] - sum_l A_l @ x[t-l-1]."""
    T, k = x_orig.shape
    e = np.zeros((T, k))
    for t in range(T):
        pred = np.zeros(k)
        for l, A in enumerate(mechanism.A_list):
            lag_t = t - l - 1
            if lag_t >= 0:
                pred += A @ x_orig[lag_t]
        e[t] = x_orig[t] - pred
    return e


def _noiseless_rollout_cf(x_orig, mechanism, t0, pert):
    """CF that is a *pure* noiseless SCM rollout from t0 (off-manifold)."""
    T, k = x_orig.shape
    x_cf = x_orig.copy()
    x_cf[t0] = x_orig[t0] + pert
    for t in range(t0 + 1, T):
        nxt = np.zeros(k)
        for l, A in enumerate(mechanism.A_list):
            lag_t = t - l - 1
            if lag_t >= 0:
                nxt += A @ x_cf[lag_t]
        x_cf[t] = nxt
    return x_cf


def _noise_reinjected_cf(x_orig, mechanism, t0, pert):
    """Pearl CF: intervene at t0, propagate while re-injecting original noise."""
    T, k = x_orig.shape
    e = _abduct(x_orig, mechanism)
    x_cf = x_orig.copy()
    x_cf[t0] = x_orig[t0] + pert
    for t in range(t0 + 1, T):
        nxt = e[t].copy()
        for l, A in enumerate(mechanism.A_list):
            lag_t = t - l - 1
            if lag_t >= 0:
                nxt += A @ x_cf[lag_t]
        x_cf[t] = nxt
    return x_cf


class TestCFFaithPearl:
    def test_invalid_semantics_raises(self):
        """Unknown semantics is rejected at construction."""
        with pytest.raises(ValueError):
            CFfaith(semantics="bogus")

    def test_identical_cf_pearl_hard_is_one(self):
        """Identical CF → delta=0 → faithful under Pearl semantics (hard=1)."""
        k, L, T = 3, 1, 20
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=5)
        x_cf = x_orig.copy()

        scorer = CFfaith(tol=1e-3, semantics="pearl_delta")
        result = scorer.score(x_orig, x_cf, 5, graph, mechanism)
        assert result["hard"] == 1.0

    def test_noise_reinjected_cf_diverges(self):
        """A Pearl CF (noise re-injected): pearl hard=1, but rollout hard=0.

        This is the core divergence — the same CF is faithful under one metric
        and unfaithful under the other. A noise-reinjected CF stays on the data
        manifold (Pearl-faithful) but is not a pure noiseless rollout.
        """
        k, L, T = 3, 1, 25
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=6)
        t0 = 8
        rng = np.random.default_rng(11)
        x_cf = _noise_reinjected_cf(x_orig, mechanism, t0, rng.uniform(-0.4, 0.4, k))

        pearl = CFfaith(tol=1e-3, semantics="pearl_delta").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        rollout = CFfaith(tol=1e-3, semantics="noiseless_rollout").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        assert pearl["hard"] == 1.0, f"pearl should accept, got {pearl}"
        assert rollout["hard"] == 0.0, f"rollout should reject, got {rollout}"

    def test_noiseless_built_cf_diverges(self):
        """A pure noiseless-rollout CF: rollout hard=1, but pearl hard=0 (mirror)."""
        k, L, T = 3, 1, 25
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=7)
        t0 = 8
        rng = np.random.default_rng(12)
        x_cf = _noiseless_rollout_cf(x_orig, mechanism, t0, rng.uniform(-0.4, 0.4, k))

        rollout = CFfaith(tol=1e-3, semantics="noiseless_rollout").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        pearl = CFfaith(tol=1e-3, semantics="pearl_delta").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        assert rollout["hard"] == 1.0, f"rollout should accept, got {rollout}"
        assert pearl["hard"] == 0.0, f"pearl should reject, got {pearl}"

    def test_retroactive_zero_in_both_modes(self):
        """A retroactive change is unfaithful under both semantics."""
        k, L, T = 3, 1, 20
        x_orig, graph, mechanism = _make_simple_scm(k=k, L=L, T=T, seed=8)
        t0 = 10
        x_cf = x_orig.copy()
        x_cf[3, 0] += 5.0  # change before t0

        for sem in CFfaith.SEMANTICS:
            result = CFfaith(tol=1e-3, semantics=sem).score(
                x_orig, x_cf, t0, graph, mechanism
            )
            assert result["hard"] == 0.0, f"{sem} should reject retroactive, got {result}"

