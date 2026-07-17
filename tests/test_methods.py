"""Tests for the three CF generators on a small trained LSTM.

Verifies output shapes/finiteness, that Wachter flips at least one label, and
that CARLA-causal has zero retroactive change and is CF-faith (rollout) hard=1
by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.methods import (
    CARLARecourse,
    PearlCARLARecourse,
    WachterCF,
    derive_intervention_t,
)
from causaltemp_xai.metrics.cf_faith import CFfaith


@pytest.fixture(scope="module")
def trained():
    """Small VAR dataset + lightly trained LSTMClassifier (shared across tests)."""
    gen = LinearSCMT(k=5, L=1, T=30, N=300, seed=0)
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    clf = LSTMClassifier(
        n_inputs=5,
        hidden_size=16,
        num_layers=2,
        dropout=0.0,
        lr=5e-3,
        batch_size=64,
        max_epochs=25,
        patience=25,
        seed=0,
    )
    clf.fit(X[:240], Y[:240], X[240:], Y[240:])
    return clf, data


# ---------------------------------------------------------------------------
# Wachter
# ---------------------------------------------------------------------------


class TestWachter:
    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = WachterCF(target_class=1, n_steps=60, lr=0.1).generate(x, clf)
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_flips_at_least_one(self, trained):
        clf, data = trained
        X = data["X"]
        preds = clf.predict(X[:20])
        # Choose a few instances NOT in class 1, ask Wachter to flip them.
        src = [i for i in range(20) if preds[i] == 0][:3] or list(range(3))
        wachter = WachterCF(target_class=1, n_steps=100, lr=0.1)
        flips = 0
        for i in src:
            cf = wachter.generate(X[i], clf)
            if clf.predict(cf) == 1:
                flips += 1
        assert flips >= 1, "Wachter failed to flip any of the selected instances"


# ---------------------------------------------------------------------------
# CARLA-causal
# ---------------------------------------------------------------------------


class TestCARLA:
    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        clf, data = trained
        x = data["X"][3]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        # Nothing before the derived intervention point may change.
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_rollout_hard_is_one(self, trained):
        """CARLA emits a noiseless rollout → rollout-faith hard=1 by construction."""
        clf, data = trained
        x = data["X"][3]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        scorer = CFfaith(tol=1e-3, semantics="noiseless_rollout")
        result = scorer.score(x, cf, t0, data["graph"], data["mechanism"])
        assert result["hard"] == 1.0, f"expected rollout hard=1, got {result}"


# ---------------------------------------------------------------------------
# Pearl-CARLA (M2, 2026-07-08) — noise-reinjecting recourse variant
# ---------------------------------------------------------------------------


class TestPearlCARLA:
    """Mirrors TestCARLA, but the invariant that holds "by construction" is
    ``cf_faith_pearl_hard == 1`` (not ``rollout_hard``) — see
    ``causaltemp_xai/methods/counterfactual/carla.py``'s module docstring for
    why the two variants differ, and the docstring of ``PearlCARLARecourse``
    for the empirical long-horizon-validity finding (not a naive "always
    recovers validity" result — smoke-scale confirmed instead: zero
    retroactive change still holds bit-identically pre-t0 because both
    variants copy the factual prefix verbatim regardless of rollout
    semantics; only the *forward* (post-t0) reference trajectory differs
    between rollout and pearl_delta semantics).
    """

    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        cf = PearlCARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        """Still holds: both CARLARecourse and PearlCARLARecourse copy
        ``x[:t0]`` verbatim into the CF regardless of forward-rollout
        semantics (only the post-t0 region differs between the two)."""
        clf, data = trained
        x = data["X"][3]
        cf = PearlCARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_pearl_hard_is_one(self, trained):
        """PearlCARLARecourse reinjects the abducted factual noise -> Pearl-faith
        hard=1 by construction (the rollout-semantics scorer should generally
        NOT also score hard=1 -- the two are structurally different CFs)."""
        clf, data = trained
        x = data["X"][3]
        cf = PearlCARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanism"]
        )
        t0 = derive_intervention_t(x, cf)
        pearl_scorer = CFfaith(tol=1e-3, semantics="pearl_delta")
        result = pearl_scorer.score(x, cf, t0, data["graph"], data["mechanism"])
        assert result["hard"] == 1.0, f"expected pearl hard=1, got {result}"

    def test_generate_batch_shape(self, trained):
        clf, data = trained
        X = data["X"][:4]
        cfs = PearlCARLARecourse(target_class=1, n_steps=40, t0_fractions=(0.5,)).generate_batch(
            X, clf, data["graph"], data["mechanism"]
        )
        assert cfs.shape == X.shape
        assert np.all(np.isfinite(cfs))
