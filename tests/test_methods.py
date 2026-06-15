"""Tests for the three CF generators on a small trained TCN.

Verifies output shapes/finiteness, that Wachter flips at least one label, and
that CARLA-causal has zero retroactive change and is CF-faith (rollout) hard=1
by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.classifiers import TCNClassifier
from causaltemp_xai.methods import CARLARecourse, DiCECF, WachterCF, derive_intervention_t
from causaltemp_xai.metrics.cf_faith import CFfaith


@pytest.fixture(scope="module")
def trained():
    """Small VAR dataset + lightly trained TCNClassifier (shared across tests)."""
    gen = LinearSCMT(k=5, L=1, T=30, N=300, seed=0)
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    clf = TCNClassifier(
        n_inputs=5,
        n_levels=2,
        n_channels=16,
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
            x, clf, data["graph"], data["mechanisms"]
        )
        assert cf.shape == x.shape
        assert np.all(np.isfinite(cf))

    def test_zero_retroactive_change(self, trained):
        clf, data = trained
        x = data["X"][3]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanisms"]
        )
        t0 = derive_intervention_t(x, cf)
        # Nothing before the derived intervention point may change.
        assert np.allclose(cf[:t0], x[:t0], atol=1e-6)

    def test_cf_faith_rollout_hard_is_one(self, trained):
        """CARLA emits a noiseless rollout → rollout-faith hard=1 by construction."""
        clf, data = trained
        x = data["X"][3]
        cf = CARLARecourse(target_class=1, n_steps=60, t0_fractions=(0.5,)).generate(
            x, clf, data["graph"], data["mechanisms"]
        )
        t0 = derive_intervention_t(x, cf)
        scorer = CFfaith(tol=1e-3, semantics="noiseless_rollout")
        result = scorer.score(x, cf, t0, data["graph"], data["mechanisms"])
        assert result["hard"] == 1.0, f"expected rollout hard=1, got {result}"


# ---------------------------------------------------------------------------
# DiCE
# ---------------------------------------------------------------------------


class TestDiCE:
    def test_fallback_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        dice = DiCECF(target_class=1, n_cfs=3, n_steps=100, use_dice_ml=False)
        cfs = dice.generate(x, clf)
        assert dice.backend_used == "fallback"
        assert cfs.shape == (3, x.shape[0], x.shape[1])
        assert np.all(np.isfinite(cfs))

    def test_dice_ml_backend_returns_cfs(self, trained):
        """The user-selected dice-ml gradient path returns correctly-shaped CFs."""
        clf, data = trained
        X = data["X"]
        bg = X[:40]
        # Pick an instance the classifier predicts as 0; flip toward 1.
        preds = clf.predict(X[:30])
        idx = next((i for i in range(30) if preds[i] == 0), 0)
        dice = DiCECF(target_class=1, n_cfs=2, background_data=bg, use_dice_ml=True)
        cfs = dice.generate(X[idx], clf)
        assert dice.backend_used == "dice-ml"
        assert cfs.ndim == 3 and cfs.shape[1:] == (X.shape[1], X.shape[2])
        assert np.all(np.isfinite(cfs))
