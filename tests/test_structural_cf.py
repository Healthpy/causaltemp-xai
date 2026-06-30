"""Tests for the oracle structural counterfactual (positive control).

Validates abduction-action-prediction on a nonlinear ``NlinearSCMT`` SCM:

- **Abduction exactness** — re-adding abducted noise to the mechanism rollout
  reconstructs the factual trajectory.
- **Positive control** — the Pearl oracle CF is ``pearl_delta``-faithful
  (hard=1); the noiseless skeleton CF is ``noiseless_rollout``-faithful.
- **Mutual exclusivity** — neither oracle is hard=1 under *both* semantics.
- **Negative controls** — retroactive and random-perturbation CFs score hard=0.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmark.generator import NlinearSCMT
from causaltemp_xai.benchmark.structural_cf import (
    _window,
    abduct_noise,
    structural_counterfactual,
)
from causaltemp_xai.metrics.cf_faith import CFfaith


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nlinear(k: int = 3, L: int = 2, T: int = 25, seed: int = 7):
    """Return (x_original, graph, mechanism) from a SMOKE-scale NlinearSCMT."""
    gen = NlinearSCMT(k=k, L=L, T=T, N=4, seed=seed, hidden=8)
    data = gen.generate(burn_in=20)
    return data["X"][0], data["graph"], data["mechanism"]


def _reconstruct(eps: np.ndarray, mechanism) -> np.ndarray:
    """Roll a trajectory forward from scratch reusing ``eps`` (no factual)."""
    T, k = eps.shape
    L = mechanism.L
    x = np.zeros((T, k))
    for t in range(T):
        x[t] = mechanism.forward_numpy(_window(x, t, L, k)) + eps[t]
    return x


# ---------------------------------------------------------------------------
# Abduction
# ---------------------------------------------------------------------------


class TestAbduction:
    def test_abduction_reconstructs_factual(self):
        """Re-adding abducted eps to the rollout reconstructs x_orig (< 1e-6)."""
        x_orig, _, mechanism = _make_nlinear()
        eps = abduct_noise(x_orig, mechanism)
        x_rec = _reconstruct(eps, mechanism)
        assert np.abs(x_rec - x_orig).max() < 1e-6


# ---------------------------------------------------------------------------
# Positive control: oracle CF is faithful by construction
# ---------------------------------------------------------------------------


class TestOraclePositiveControl:
    def test_pearl_oracle_is_pearl_faithful(self):
        """The (noisy) oracle CF scores pearl_delta hard=1 by construction."""
        x_orig, graph, mechanism = _make_nlinear()
        t0, node = 6, 0
        value = x_orig[t0, node] + 0.5
        x_cf = structural_counterfactual(x_orig, mechanism, t0, node, value)

        r = CFfaith(tol=1e-4, semantics="pearl_delta").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        assert r["hard"] == 1.0, f"pearl oracle should be faithful, got {r}"

    def test_noiseless_oracle_is_rollout_faithful(self):
        """The noiseless skeleton CF scores noiseless_rollout hard=1."""
        x_orig, graph, mechanism = _make_nlinear()
        t0, node = 6, 1
        value = x_orig[t0, node] + 0.5
        x_cf = structural_counterfactual(
            x_orig, mechanism, t0, node, value, noiseless=True
        )

        r = CFfaith(tol=1e-4, semantics="noiseless_rollout").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        assert r["hard"] == 1.0, f"noiseless oracle should be rollout-faithful, got {r}"


# ---------------------------------------------------------------------------
# Mutual exclusivity on nonlinear data
# ---------------------------------------------------------------------------


class TestMutualExclusivity:
    def test_pearl_oracle_not_rollout_faithful(self):
        """A Pearl oracle CF is NOT a noiseless rollout (hard=0 there)."""
        x_orig, graph, mechanism = _make_nlinear()
        t0, node = 6, 0
        value = x_orig[t0, node] + 0.5
        x_cf = structural_counterfactual(x_orig, mechanism, t0, node, value)

        pearl = CFfaith(tol=1e-4, semantics="pearl_delta").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        rollout = CFfaith(tol=1e-4, semantics="noiseless_rollout").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        assert pearl["hard"] == 1.0
        assert rollout["hard"] == 0.0, f"pearl oracle must not be rollout-faithful, {rollout}"

    def test_noiseless_oracle_not_pearl_faithful(self):
        """A noiseless skeleton CF is NOT Pearl-faithful (hard=0 there)."""
        x_orig, graph, mechanism = _make_nlinear()
        t0, node = 6, 1
        value = x_orig[t0, node] + 0.5
        x_cf = structural_counterfactual(
            x_orig, mechanism, t0, node, value, noiseless=True
        )

        rollout = CFfaith(tol=1e-4, semantics="noiseless_rollout").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        pearl = CFfaith(tol=1e-4, semantics="pearl_delta").score(
            x_orig, x_cf, t0, graph, mechanism
        )
        assert rollout["hard"] == 1.0
        assert pearl["hard"] == 0.0, f"noiseless oracle must not be pearl-faithful, {pearl}"


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


class TestNegativeControls:
    def test_retroactive_cf_both_hard_zero(self):
        """A retroactively-edited oracle CF scores hard=0 under both semantics."""
        x_orig, graph, mechanism = _make_nlinear()
        t0, node = 8, 0
        value = x_orig[t0, node] + 0.5
        x_cf = structural_counterfactual(x_orig, mechanism, t0, node, value)
        x_cf[t0 - 3, 0] += 5.0  # retroactive change before t0

        for sem in CFfaith.SEMANTICS:
            r = CFfaith(tol=1e-4, semantics=sem).score(
                x_orig, x_cf, t0, graph, mechanism
            )
            assert r["hard"] == 0.0, f"{sem} should reject retroactive, got {r}"

    def test_random_perturbation_cf_unfaithful(self):
        """A random post-t0 perturbation scores hard=0 and soft<1 under both."""
        x_orig, graph, mechanism = _make_nlinear()
        t0 = 6
        rng = np.random.default_rng(123)
        x_cf = x_orig.copy()
        x_cf[t0:] += rng.uniform(-0.5, 0.5, x_orig[t0:].shape)

        for sem in CFfaith.SEMANTICS:
            r = CFfaith(tol=1e-4, semantics=sem).score(
                x_orig, x_cf, t0, graph, mechanism
            )
            assert r["hard"] == 0.0, f"{sem} random CF should be hard=0, got {r}"
            assert r["soft"] < 1.0, f"{sem} random CF should be soft<1, got {r}"
