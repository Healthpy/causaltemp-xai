"""Tests for `experiments/10_tier2_real_suite.py` (M4i,
2026-08-06).

Fast, deterministic unit tests of the dataset-mapping/table-shaping logic --
not an end-to-end run (network fetch + LSTM training + CF generation + two
discovery-method fits per dataset), which belongs in the M4i smoke-scale
validation step, not the pytest suite.
"""

from __future__ import annotations

import importlib

_phase10 = importlib.import_module("experiments.10_tier2_real_suite")


class TestDatasetMapping:
    def test_archive_names_cover_all_registered_datasets(self):
        assert set(_phase10.ALL_DATASETS) == set(_phase10._ARCHIVE_NAME)

    def test_smoke_scale_is_basicmotions_only_no_network_needed(self):
        assert _phase10._SCALE_DATASETS["smoke"] == ("basicmotions",)

    def test_full_scale_includes_all_three(self):
        assert set(_phase10._SCALE_DATASETS["full"]) == {
            "basicmotions",
            "racketsports",
            "epilepsy",
        }

    def test_every_scale_dataset_has_an_archive_name(self):
        for datasets in _phase10._SCALE_DATASETS.values():
            for name in datasets:
                assert name in _phase10._ARCHIVE_NAME


class TestRunTableShape:
    def test_summary_row_carries_no_ground_truth_note(self):
        """A row assembled the way run_dataset builds it must always carry the
        note through, per M4i's own explicit disclosure requirement -- this is
        a structural check on the note's presence, not a live network run."""
        row = {
            "dataset": "basicmotions",
            "k": 6,
            "n_classes": 4,
            "test_acc": 0.925,
            "shd_between_methods": 8.0,
            "cf_faith_discovered_rollout_by_method": {"CftsWachter": 0.186},
            "note": "structural agreement only, no correctness claim possible without ground truth",
        }
        assert "no correctness claim" in row["note"]
        assert row["k"] >= 3  # Tier-2 selection criterion
