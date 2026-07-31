"""Tests for Phase 08's horizon-sweep grid (M4d).

The sweep's whole job is to report validity and CF-faith *as a function of*
``T - t0``. Its correctness is therefore mostly about the grid: a horizon that
is silently substituted, dropped without notice, or resolved to a different
``t0`` than the one it is labelled with produces a fabricated point on the
decay curve that no downstream check would catch — the numbers would be real,
just attributed to the wrong horizon.

Phase 08 is a numbered-prefix module, loaded the way
``tests/test_experiment_registry.py`` already loads Phase 03.
"""

from __future__ import annotations

import importlib

import pytest

_phase08 = importlib.import_module("experiments.08_horizon_sweep")
resolve_horizons = _phase08.resolve_horizons


class TestResolveHorizons:
    def test_explicit_spec_is_honoured_exactly(self):
        assert resolve_horizons(100, "2,10,50") == [2, 10, 50]

    def test_spec_is_sorted_and_deduplicated(self):
        """The curve is plotted in horizon order; duplicates would double-weight
        a point in any downstream fit."""
        assert resolve_horizons(100, "50,2,10,50, 2") == [2, 10, 50]

    def test_default_grid_is_denser_near_the_knee(self):
        """Geometric decay means the curve is flat at large horizons. If the
        grid were uniform the collapse knee -- the only part that carries the
        claim -- would be resolved by one or two points. The first `full`
        validation run put the knee between h=10 and h=50, and `full_nl` decays
        ~4 orders faster, so most points must sit in the lower quarter."""
        got = resolve_horizons(100, None)
        assert got == sorted(got)
        below = [h for h in got if h <= 25]
        assert len(below) > len(got) / 2, f"grid not knee-dense: {got}"

    def test_default_grid_is_roughly_geometric(self):
        """The principled grid for a geometric decay is geometric. Guards
        against someone 'tidying' the constant into a uniform range, which
        would look neater and resolve the knee worse."""
        got = resolve_horizons(100, None)
        ratios = [b / a for a, b in zip(got, got[1:])]
        assert min(ratios) > 1.0, f"non-increasing grid: {got}"
        # A uniform grid's ratios shrink monotonically toward 1 at the top end;
        # a geometric one keeps them bounded away from 1 throughout.
        assert sum(r > 1.2 for r in ratios) >= len(ratios) - 2, f"grid too uniform: {got}"

    def test_default_grid_brackets_the_locked_operating_point(self):
        """The committed runs use t0 in {25, 50} of T=100, i.e. horizons
        {75, 50}. The sweep must place points on both sides of those so the
        published operating point can be located on the curve."""
        got = resolve_horizons(100, None)
        for locked in (50, 75):
            assert min(got) < locked < max(got)

    def test_unscorable_horizons_are_dropped_with_a_notice(self, capsys):
        """h == T means t0 == 0 (no prefix, abduction undefined) and h == 1
        means t0 == T-1 (CF-faith degeneracy gate). Both must go, and the drop
        must be announced -- a silently shorter curve looks like a complete one."""
        got = resolve_horizons(100, "1,2,50,100,200")
        assert got == [2, 50]
        assert "dropped unscorable horizons" in capsys.readouterr().out

    def test_all_unscorable_is_fatal_rather_than_empty(self):
        with pytest.raises(SystemExit, match="no scorable horizon"):
            resolve_horizons(100, "1,100,500")

    def test_horizon_maps_to_the_t0_it_claims(self):
        """t0 = T - h is the definition the output columns are labelled with;
        an off-by-one here mislabels every row in per_instance.csv."""
        T = 100
        for h in resolve_horizons(T, None):
            t0 = T - h
            assert 0 < t0 < T - 1

    def test_grid_scales_with_T(self):
        """Fractional defaults must resolve against the config's own T, not a
        hardcoded scale -- smoke (T=30) and full (T=100) both use this path."""
        assert max(resolve_horizons(30, None)) < 30
        assert max(resolve_horizons(100, None)) > max(resolve_horizons(30, None))
