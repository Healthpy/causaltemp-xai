"""Tests for causaltemp_xai.stats — bootstrap CI helpers (M2, O2).

Verifies: determinism given a fixed resampling seed, sane CI ordering
(lo <= mean <= hi), narrower CIs for lower-variance data, correct NaN
handling, degenerate (n=1 / empty) edge cases, and that the hierarchical
(seed-cluster) bootstrap behaves sensibly relative to the flat bootstrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.stats import bootstrap_ci, collapse_horizon_ci, hierarchical_bootstrap_ci

# ---------------------------------------------------------------------------
# bootstrap_ci (flat, i.i.d.)
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_deterministic_given_same_seed(self):
        rng = np.random.default_rng(1)
        values = rng.normal(0.7, 0.1, size=40)
        r1 = bootstrap_ci(values, n_boot=500, seed=42)
        r2 = bootstrap_ci(values, n_boot=500, seed=42)
        assert r1 == r2

    def test_ci_ordering(self):
        rng = np.random.default_rng(2)
        values = rng.normal(0.5, 0.2, size=50)
        r = bootstrap_ci(values, n_boot=1000, seed=0)
        assert r.ci_lo <= r.mean <= r.ci_hi

    def test_narrower_ci_for_lower_variance(self):
        rng = np.random.default_rng(3)
        low_var = rng.normal(0.9, 0.01, size=30)
        high_var = rng.normal(0.9, 0.4, size=30)
        r_low = bootstrap_ci(low_var, n_boot=2000, seed=0)
        r_high = bootstrap_ci(high_var, n_boot=2000, seed=0)
        assert (r_low.ci_hi - r_low.ci_lo) < (r_high.ci_hi - r_high.ci_lo)

    def test_constant_data_gives_point_ci(self):
        values = np.ones(20)
        r = bootstrap_ci(values, n_boot=500, seed=0)
        assert r.mean == pytest.approx(1.0)
        assert r.ci_lo == pytest.approx(1.0)
        assert r.ci_hi == pytest.approx(1.0)

    def test_nan_entries_dropped(self):
        values = [1.0, 1.0, np.nan, 1.0, np.nan]
        r = bootstrap_ci(values, n_boot=200, seed=0)
        assert r.n == 3
        assert r.mean == pytest.approx(1.0)

    def test_all_nan_returns_nan_result(self):
        values = [np.nan, np.nan]
        r = bootstrap_ci(values, n_boot=200, seed=0)
        assert r.n == 0
        assert np.isnan(r.mean)
        assert np.isnan(r.ci_lo)
        assert np.isnan(r.ci_hi)

    def test_single_value_gives_point_ci(self):
        r = bootstrap_ci([0.42], n_boot=200, seed=0)
        assert r.n == 1
        assert r.mean == pytest.approx(0.42)
        assert r.ci_lo == pytest.approx(0.42)
        assert r.ci_hi == pytest.approx(0.42)

    def test_empty_returns_nan_result(self):
        r = bootstrap_ci([], n_boot=200, seed=0)
        assert r.n == 0
        assert np.isnan(r.mean)

    def test_as_dict_prefix(self):
        r = bootstrap_ci([1.0, 2.0, 3.0], n_boot=100, seed=0)
        d = r.as_dict(prefix="validity_")
        assert set(d.keys()) == {"validity_mean", "validity_ci_lo", "validity_ci_hi", "validity_n"}


# ---------------------------------------------------------------------------
# hierarchical_bootstrap_ci (seed-cluster)
# ---------------------------------------------------------------------------


class TestHierarchicalBootstrapCI:
    def test_deterministic_given_same_seed(self):
        rng = np.random.default_rng(4)
        groups = [rng.normal(0.8, 0.1, size=10) for _ in range(5)]
        r1 = hierarchical_bootstrap_ci(groups, n_boot=500, seed=7)
        r2 = hierarchical_bootstrap_ci(groups, n_boot=500, seed=7)
        assert r1 == r2

    def test_ci_ordering(self):
        rng = np.random.default_rng(5)
        groups = [rng.normal(0.6, 0.15, size=12) for _ in range(4)]
        r = hierarchical_bootstrap_ci(groups, n_boot=1000, seed=0)
        assert r.ci_lo <= r.mean <= r.ci_hi

    def test_pools_total_n_across_seeds(self):
        groups = [np.ones(10), np.ones(20) * 0.5, np.ones(15)]
        r = hierarchical_bootstrap_ci(groups, n_boot=200, seed=0)
        assert r.n == 45  # 10 + 20 + 15

    def test_between_seed_variability_widens_ci(self):
        """Seeds with wildly different per-seed means (but each internally
        constant) should give a CI *wider* than perfectly homogeneous seeds --
        this is exactly the between-seed variability a flat pooled bootstrap
        over the same total data would miss."""
        homogeneous = [np.full(10, 0.8) for _ in range(5)]
        heterogeneous = [np.full(10, m) for m in (0.2, 0.4, 0.8, 1.0, 1.0)]
        r_homo = hierarchical_bootstrap_ci(homogeneous, n_boot=2000, seed=0)
        r_hetero = hierarchical_bootstrap_ci(heterogeneous, n_boot=2000, seed=0)
        assert (r_homo.ci_hi - r_homo.ci_lo) < (r_hetero.ci_hi - r_hetero.ci_lo)

    def test_constant_across_all_seeds_gives_point_ci(self):
        groups = [np.full(8, 1.0) for _ in range(5)]
        r = hierarchical_bootstrap_ci(groups, n_boot=300, seed=0)
        assert r.mean == pytest.approx(1.0)
        assert r.ci_lo == pytest.approx(1.0)
        assert r.ci_hi == pytest.approx(1.0)

    def test_empty_groups_dropped(self):
        groups = [np.array([]), np.full(5, 0.9), np.array([np.nan, np.nan])]
        r = hierarchical_bootstrap_ci(groups, n_boot=200, seed=0)
        assert r.n == 5
        assert r.mean == pytest.approx(0.9)

    def test_all_empty_returns_nan_result(self):
        groups = [np.array([]), np.array([np.nan])]
        r = hierarchical_bootstrap_ci(groups, n_boot=100, seed=0)
        assert r.n == 0
        assert np.isnan(r.mean)

    def test_single_group_single_value_gives_point_ci(self):
        r = hierarchical_bootstrap_ci([[0.3]], n_boot=100, seed=0)
        assert r.n == 1
        assert r.mean == pytest.approx(0.3)
        assert r.ci_lo == pytest.approx(0.3)
        assert r.ci_hi == pytest.approx(0.3)

    def test_single_group_matches_flat_bootstrap_scale(self):
        """With exactly one seed group, the hierarchical bootstrap degenerates
        to resampling within that one group every iteration (the 'choose
        which seed' step has only one option) -- so its CI width should be
        comparable to (not wildly different from) the flat bootstrap's."""
        rng = np.random.default_rng(6)
        values = rng.normal(0.5, 0.2, size=40)
        r_flat = bootstrap_ci(values, n_boot=3000, seed=0)
        r_hier = hierarchical_bootstrap_ci([values], n_boot=3000, seed=0)
        assert r_flat.mean == pytest.approx(r_hier.mean)
        # Widths should be within a generous factor of each other (both are
        # stochastic estimates of the same underlying single-group spread).
        width_flat = r_flat.ci_hi - r_flat.ci_lo
        width_hier = r_hier.ci_hi - r_hier.ci_lo
        assert width_hier == pytest.approx(width_flat, rel=0.5)


# ---------------------------------------------------------------------------
# collapse_horizon_ci (M4d)
# ---------------------------------------------------------------------------


class TestCollapseHorizonCI:
    def test_exact_grid_point_crossing(self):
        """A curve that lands exactly on the threshold at a grid point should
        report that horizon without needing interpolation."""
        horizons = [5, 10, 20, 40]
        seeds = [[1.0, 1.0, 0.5, 0.0]]
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=10)
        assert r.horizon == pytest.approx(20.0)

    def test_interpolates_between_grid_points(self):
        """Threshold crossed strictly between two tested horizons -> linear
        interpolation, not a snap to the nearest grid point."""
        horizons = [0, 10]
        seeds = [[1.0, 0.0]]  # straight line, crosses 0.5 exactly at x=5
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=10)
        assert r.horizon == pytest.approx(5.0)

    def test_never_collapses_is_nan_not_extrapolated(self):
        """Validity stays above threshold at every tested horizon -- the
        honest answer is 'no evidence of a crossing in this range', not a
        fabricated point beyond the last horizon tested."""
        horizons = [5, 10, 20]
        seeds = [[1.0, 0.9, 0.8]]
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=10)
        assert np.isnan(r.horizon)
        assert r.frac_boot_crossed == pytest.approx(0.0)

    def test_already_collapsed_at_first_horizon_is_nan_not_first_point(self):
        """Validity is already at/below threshold at the smallest horizon
        tested -- the true crossing may lie before it, which this grid
        cannot see, so this must not silently report horizons[0]."""
        horizons = [5, 10, 20]
        seeds = [[0.4, 0.3, 0.1]]
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=10)
        assert np.isnan(r.horizon)

    def test_deterministic_given_same_seed(self):
        rng = np.random.default_rng(1)
        horizons = [5, 10, 20, 40]
        seeds = [
            np.clip(1.0 - rng.normal(0.02, 0.005) * np.array(horizons), 0, 1) for _ in range(5)
        ]
        r1 = collapse_horizon_ci(horizons, seeds, n_boot=500, seed=3)
        r2 = collapse_horizon_ci(horizons, seeds, n_boot=500, seed=3)
        assert r1 == r2

    def test_ci_ordering_when_crossing_found(self):
        rng = np.random.default_rng(2)
        horizons = [5, 10, 20, 40, 80]
        # Decaying curves with seed-level noise, guaranteed to cross 0.5.
        seeds = [
            np.clip(1.2 - 0.02 * np.array(horizons) + rng.normal(0, 0.05, 5), 0, 1)
            for _ in range(6)
        ]
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=2000, seed=0)
        if r.frac_boot_crossed > 0:
            assert r.ci_lo <= r.horizon <= r.ci_hi

    def test_single_seed_gives_point_ci(self):
        horizons = [5, 10, 20]
        seeds = [[1.0, 0.6, 0.2]]
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=500)
        assert r.n_seeds == 1
        assert r.ci_lo == r.horizon == r.ci_hi

    def test_frac_boot_crossed_reflects_marginal_curves(self):
        """A curve that hovers right at the threshold should cross in only
        some bootstrap resamples, not all -- frac_boot_crossed must reflect
        that rather than silently reporting a full-confidence interval."""
        horizons = [5, 10, 20]
        rng = np.random.default_rng(9)
        seeds = [
            [0.5 + rng.normal(0, 0.05), 0.5 + rng.normal(0, 0.05), 0.5 + rng.normal(0, 0.05)]
            for _ in range(8)
        ]
        r = collapse_horizon_ci(horizons, seeds, threshold=0.5, n_boot=2000, seed=0)
        assert 0.0 < r.frac_boot_crossed < 1.0
