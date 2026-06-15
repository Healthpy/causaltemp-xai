"""Tests for derive_intervention_t — the uniform intervention-time rule."""

from __future__ import annotations

import numpy as np

from causaltemp_xai.methods import derive_intervention_t


def _base(T=10, k=3):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, k))


class TestDeriveInterventionT:
    def test_first_changed_timestep(self):
        x = _base()
        x_cf = x.copy()
        x_cf[4, 1] += 1.0  # first change at t=4
        assert derive_intervention_t(x, x_cf) == 4

    def test_earliest_change_wins(self):
        x = _base()
        x_cf = x.copy()
        x_cf[7, 0] += 2.0
        x_cf[3, 2] += 0.5  # earlier change at t=3 should dominate
        assert derive_intervention_t(x, x_cf) == 3

    def test_all_equal_returns_T_minus_1(self):
        x = _base(T=12)
        x_cf = x.copy()
        assert derive_intervention_t(x, x_cf) == 11

    def test_change_at_t0(self):
        x = _base()
        x_cf = x.copy()
        x_cf[0, 0] += 5.0
        assert derive_intervention_t(x, x_cf) == 0

    def test_change_at_last_timestep(self):
        x = _base(T=8)
        x_cf = x.copy()
        x_cf[7, 0] += 1.0
        assert derive_intervention_t(x, x_cf) == 7

    def test_tol_ignores_subthreshold_changes(self):
        x = _base()
        x_cf = x.copy()
        x_cf[2, 0] += 1e-9      # below tol → ignored
        x_cf[6, 1] += 1.0       # above tol → detected
        assert derive_intervention_t(x, x_cf, tol=1e-6) == 6
