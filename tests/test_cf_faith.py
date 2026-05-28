"""Tests for CFfaith — causal-faithfulness scorer.

Verifies:
- Perfectly SCM-compliant CF → hard=1.0, soft close to 1.0.
- CF with retroactive change (before intervention_t) → hard=0.0, soft=0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.metrics.cf_faith import CFfaith


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_scm(k: int = 3, L: int = 1, T: int = 20, seed: int = 0):
    """Return (x_original, graph, mechanisms) for a simple VAR(L) system."""
    rng = np.random.default_rng(seed)
    # Random stable coefficient matrices
    mechanisms = []
    for _ in range(L):
        A = rng.uniform(-0.3, 0.3, (k, k))
        mechanisms.append(A)

    graph = np.stack([(A != 0).astype(float) for A in mechanisms], axis=-1)  # (k, k, L)

    # Simulate T time steps
    x = np.zeros((T, k))
    noise = rng.laplace(0, 0.01, (T, k))
    for t in range(L, T):
        for lag, A in enumerate(mechanisms, start=1):
            x[t] += A @ x[t - lag]
        x[t] += noise[t]

    return x, graph, mechanisms


# ---------------------------------------------------------------------------
# SCM-compliant CF
# ---------------------------------------------------------------------------


class TestCFFaithCompliant:
    def test_hard_score_is_one(self):
        """CF exactly obeying the SCM forward from intervention_t → hard=1."""
        k, L, T = 4, 1, 30
        x_orig, graph, mechanisms = _make_simple_scm(k=k, L=L, T=T, seed=1)
        intervention_t = 10
        rng = np.random.default_rng(42)

        # Build a CF: identical up to intervention_t, then perturb at intervention_t
        # and propagate forward strictly via the SCM mechanisms.
        x_cf = x_orig.copy()
        x_cf[intervention_t] += rng.uniform(-0.5, 0.5, k)  # intervention

        # Propagate exactly via mechanisms (no noise)
        for t in range(intervention_t + 1, T):
            x_cf[t] = np.zeros(k)
            for lag, A in enumerate(mechanisms, start=1):
                if t - lag >= 0:
                    x_cf[t] += A @ x_cf[t - lag]

        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanisms)
        assert result["hard"] == 1.0, f"Expected hard=1.0, got {result['hard']}"

    def test_soft_score_near_one(self):
        """SCM-compliant CF → soft score should be close to 1.0."""
        k, L, T = 3, 1, 25
        x_orig, graph, mechanisms = _make_simple_scm(k=k, L=L, T=T, seed=2)
        intervention_t = 8
        rng = np.random.default_rng(7)

        x_cf = x_orig.copy()
        x_cf[intervention_t] += rng.uniform(-0.3, 0.3, k)
        for t in range(intervention_t + 1, T):
            x_cf[t] = np.zeros(k)
            for lag, A in enumerate(mechanisms, start=1):
                if t - lag >= 0:
                    x_cf[t] += A @ x_cf[t - lag]

        scorer = CFfaith(tol=1e-3, scale=1.0)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanisms)
        assert result["soft"] > 0.9, f"Expected soft>0.9, got {result['soft']}"


# ---------------------------------------------------------------------------
# Retroactive CF
# ---------------------------------------------------------------------------


class TestCFFaithRetroactive:
    def test_hard_score_is_zero(self):
        """CF with a retroactive change (before intervention_t) → hard=0.0."""
        k, L, T = 4, 1, 30
        x_orig, graph, mechanisms = _make_simple_scm(k=k, L=L, T=T, seed=3)
        intervention_t = 15

        x_cf = x_orig.copy()
        # Retroactive: modify a time step BEFORE intervention_t
        x_cf[5] += 999.0  # blatant change before intervention

        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanisms)
        assert result["hard"] == 0.0, f"Expected hard=0.0, got {result['hard']}"

    def test_soft_score_is_zero_on_retroactive(self):
        """Retroactive CF → soft=0.0 (early-return path)."""
        k, L, T = 3, 1, 20
        x_orig, graph, mechanisms = _make_simple_scm(k=k, L=L, T=T, seed=4)
        intervention_t = 10

        x_cf = x_orig.copy()
        x_cf[3, 0] += 5.0  # retroactive change

        scorer = CFfaith(tol=1e-4)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanisms)
        assert result["soft"] == 0.0, f"Expected soft=0.0, got {result['soft']}"

    def test_identical_cf_hard_is_one(self):
        """Identical CF with zero intervention → forward residual is 0 → hard=1."""
        k, L, T = 3, 1, 20
        x_orig, graph, mechanisms = _make_simple_scm(k=k, L=L, T=T, seed=5)
        intervention_t = 5
        x_cf = x_orig.copy()

        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_orig, x_cf, intervention_t, graph, mechanisms)
        assert result["hard"] == 1.0

