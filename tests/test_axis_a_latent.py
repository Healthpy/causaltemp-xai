"""Minimal regression tests for the Axis-A latent branch (fix #9, 2026-07-18).

Each test pins one of the defects fixed on 2026-07-18 so it cannot return:

* ``mcc`` used |Pearson| while claiming invariance to **monotone**
  reparameterisation (which only a rank correlation provides) — now Spearman.
* ``mig`` discretised the MI numerator and the entropy denominator with two
  *different* binnings, so ``I(z;v) <= H(v)`` was not guaranteed and the
  documented ``[0, 1]`` range could be exceeded — now one shared binning.
* ``dci`` crossed its normalisers (``D_j`` by ``ln(d_z)`` instead of ``ln(K)``
  and vice versa; measured ``D = -0.499`` at ``d_z=2, K=4``) and scored dead
  latents 1.0, averaging them in unweighted — now per-support normalisers and
  importance-weighted D.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.metrics.axis_a import dci, mcc, mig

FAST_REG = {"n_estimators": 10, "max_depth": 2}


def _rng():
    return np.random.default_rng(0)


class TestMCCMonotoneInvariance:
    def test_perfect_recovery_is_one(self):
        V = _rng().normal(size=(300, 3))
        assert mcc(V.copy(), V) == pytest.approx(1.0)

    def test_monotone_reparameterisation_invariance(self):
        """exp() is monotone: a perfectly recovered latent must still score 1.
        |Pearson| broke this (the pre-fix defect); |Spearman| holds it."""
        V = _rng().normal(size=(300, 3))
        Z = np.exp(V)  # elementwise monotone transform of the true factors
        assert mcc(Z, V) == pytest.approx(1.0)

    def test_independent_latents_score_low(self):
        rng = _rng()
        V = rng.normal(size=(300, 3))
        Z = rng.normal(size=(300, 3))  # unrelated
        assert mcc(Z, V) < 0.3


class TestMIGRange:
    def test_within_documented_range(self):
        """Shared binning restores I <= H, so MIG stays in [0, 1] — the
        pre-fix mixed binnings could push it above 1."""
        rng = _rng()
        V = rng.normal(size=(500, 2))
        Z = np.column_stack([V[:, 0], rng.normal(size=500)])
        score = mig(Z, V)
        assert 0.0 <= score <= 1.0

    def test_dedicated_latent_beats_shared(self):
        rng = _rng()
        V = rng.normal(size=(500, 2))
        Z_dedicated = V + 0.05 * rng.normal(size=V.shape)
        shared = V[:, 0] + V[:, 1]
        Z_shared = np.column_stack([shared, shared + 0.05 * rng.normal(size=500)])
        assert mig(Z_dedicated, V) > mig(Z_shared, V)


class TestDCIFixes:
    def test_scores_in_range_for_dz_neq_K(self):
        """Crossed normalisers pushed D to -0.499 at d_z=2, K=4."""
        rng = _rng()
        V = rng.normal(size=(300, 4))
        Z = V[:, :2] + 0.1 * rng.normal(size=(300, 2))  # d_z=2 != K=4
        out = dci(Z, V, regressor_kwargs=FAST_REG)
        assert 0.0 <= out["disentanglement"] <= 1.0
        assert 0.0 <= out["completeness"] <= 1.0

    def test_dead_latents_do_not_inflate_disentanglement(self):
        """A representation padded with constant (dead) latents must not
        outscore the same representation without them — pre-fix the dead
        columns scored D_j = 1.0 each and were averaged in."""
        rng = _rng()
        V = rng.normal(size=(300, 2))
        Z_live = V + 0.1 * rng.normal(size=V.shape)
        Z_padded = np.column_stack([Z_live, np.zeros((300, 6))])  # 75% dead
        d_live = dci(Z_live, V, regressor_kwargs=FAST_REG)["disentanglement"]
        d_padded = dci(Z_padded, V, regressor_kwargs=FAST_REG)["disentanglement"]
        assert d_padded <= d_live + 0.05  # no free credit for dead latents
