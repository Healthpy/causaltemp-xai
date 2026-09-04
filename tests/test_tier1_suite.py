"""Tests for `experiments/09_tier1_synthetic_suite.py` (M4i,
2026-08-06).

Fast, deterministic unit tests of the pure helper logic (job resolution,
seed-pooling) -- not an end-to-end run, which is expensive (LSTM training +
CF generation + two discovery-method fits per config, per seed) and belongs
in the M4i smoke-scale validation step, not the pytest suite.
"""

from __future__ import annotations

import importlib

import pytest

_phase09 = importlib.import_module("experiments.09_tier1_synthetic_suite")


class TestResolveJobs:
    def test_all_three_families_smoke_all_resolve(self):
        jobs, skipped = _phase09._resolve_jobs(list(_phase09.ALL_FAMILIES), ["smoke"])
        assert len(jobs) == 3
        assert skipped == []
        resolved = {family: config for family, _scale, config in jobs}
        assert resolved == {
            "linear": "smoke",
            "mlp": "smoke_nl",
            "spring": "smoke_spring",
        }

    def test_full_scale_includes_spring(self):
        jobs, skipped = _phase09._resolve_jobs(list(_phase09.ALL_FAMILIES), ["full"])
        resolved = {family: config for family, _scale, config in jobs}
        assert resolved == {
            "linear": "full",
            "mlp": "full_nl",
            "spring": "full_spring",
        }
        assert skipped == []

    def test_both_scales_yields_six_jobs(self):
        jobs, skipped = _phase09._resolve_jobs(list(_phase09.ALL_FAMILIES), ["smoke", "full"])
        assert len(jobs) == 6
        assert skipped == []

    def test_empty_job_list_raises(self):
        with pytest.raises(SystemExit):
            _phase09._resolve_jobs([], ["full"])

    def test_maskable_families_exclude_linear(self):
        assert "linear" not in _phase09._MASKABLE_FAMILIES
        assert set(_phase09._MASKABLE_FAMILIES) == {"mlp", "spring"}


class TestPoolSuitability:
    def test_pools_one_row_per_family_scale_method(self):
        records = [
            {
                "family": "mlp",
                "scale": "smoke",
                "dynotears_auc": 0.9,
                "pcmciplus_auc": 0.8,
                "dynotears_shd_to_true": 2.0,
                "pcmciplus_shd_to_true": 4.0,
                "shd_between_methods": 3.0,
            },
            {
                "family": "mlp",
                "scale": "smoke",
                "dynotears_auc": 0.92,
                "pcmciplus_auc": 0.82,
                "dynotears_shd_to_true": 1.0,
                "pcmciplus_shd_to_true": 3.0,
                "shd_between_methods": 2.0,
            },
        ]
        rows = _phase09._pool_suitability(records)
        assert len(rows) == 2  # dynotears + pcmciplus, one family/scale
        by_method = {r["method"]: r for r in rows}
        assert set(by_method) == {"dynotears", "pcmciplus"}
        assert by_method["dynotears"]["n_seeds"] == 2
        assert by_method["dynotears"]["auc_mean"] == pytest.approx(0.91, abs=1e-6)
        assert by_method["dynotears"]["shd_to_true_mean"] == pytest.approx(1.5)
        assert by_method["dynotears"]["mean_shd_between_methods"] == pytest.approx(2.5)

    def test_ci_ordering_is_sane(self):
        records = [
            {
                "family": "linear",
                "scale": "smoke",
                "dynotears_auc": a,
                "pcmciplus_auc": a - 0.05,
                "dynotears_shd_to_true": 1.0,
                "pcmciplus_shd_to_true": 2.0,
                "shd_between_methods": 1.0,
            }
            for a in (0.85, 0.90, 0.95)
        ]
        rows = _phase09._pool_suitability(records)
        for r in rows:
            assert r["auc_ci_lo"] <= r["auc_mean"] <= r["auc_ci_hi"]

    def test_empty_records_yields_no_rows(self):
        assert _phase09._pool_suitability([]) == []
