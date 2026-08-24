"""Tests for the Phase-06 provenance-audit helpers (added 2026-07-31).

Covers ``experiments._common.read_run_summary_provenance``,
``check_provenance``, and ``print_provenance_warning`` — added because the
figures/seeds reports in ``08_aggregate_and_report.py`` read only
``per_instance.csv`` (no provenance columns), so a ``git_dirty`` flag honestly
stamped into a sibling ``summary.json`` by ``dump_json`` was reachable
per-file but never surfaced at the point where publication artifacts are
produced. See ``docs/risk_register.md`` RISK-13.
"""

from __future__ import annotations

import importlib
import json

_common = importlib.import_module("experiments._common")


def _write_summary(dir_path, **fields):
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "summary.json", "w") as fh:
        json.dump(fields, fh)


class TestReadRunSummaryProvenance:
    def test_missing_file_returns_none(self, tmp_path):
        assert _common.read_run_summary_provenance(tmp_path / "summary.json") is None

    def test_reads_seed_commit_dirty(self, tmp_path):
        _write_summary(
            tmp_path,
            seed=42,
            git_commit="abc123",
            git_dirty=False,
            git_worktree_dirty=True,
        )
        prov = _common.read_run_summary_provenance(tmp_path / "summary.json")
        assert prov == {
            "seed": 42,
            "git_commit": "abc123",
            "git_dirty": False,
            "git_worktree_dirty": True,
        }

    def test_malformed_json_returns_none_not_raise(self, tmp_path):
        (tmp_path / "summary.json").write_text("{not valid json")
        assert _common.read_run_summary_provenance(tmp_path / "summary.json") is None

    def test_non_dict_json_returns_none(self, tmp_path):
        (tmp_path / "summary.json").write_text("[1, 2, 3]")
        assert _common.read_run_summary_provenance(tmp_path / "summary.json") is None

    def test_missing_fields_come_back_as_none(self, tmp_path):
        _write_summary(tmp_path, some_other_key=1)
        prov = _common.read_run_summary_provenance(tmp_path / "summary.json")
        assert prov == {
            "seed": None,
            "git_commit": None,
            "git_dirty": None,
            "git_worktree_dirty": None,
        }


class TestCheckProvenance:
    def test_clean_run(self, tmp_path):
        d = tmp_path / "full" / "lstm"
        _write_summary(d, seed=42, git_commit="deadbeef", git_dirty=False)
        report = _common.check_provenance([d])
        assert report == [
            {
                "dir": str(d),
                "status": "clean",
                "seed": 42,
                "git_commit": "deadbeef",
                "git_dirty": False,
                "git_worktree_dirty": None,
            }
        ]

    def test_dirty_run_flagged(self, tmp_path):
        d = tmp_path / "full" / "lstm"
        _write_summary(d, seed=42, git_commit="d707aae7", git_dirty=True)
        report = _common.check_provenance([d])
        assert report[0]["status"] == "dirty"

    def test_no_summary_json_flagged_missing(self, tmp_path):
        d = tmp_path / "full" / "lstm"
        d.mkdir(parents=True)
        report = _common.check_provenance([d])
        assert report[0]["status"] == "missing"

    def test_no_commit_hash_flagged_missing_even_with_summary(self, tmp_path):
        # e.g. git was unavailable when the phase ran (git_provenance() -> None)
        d = tmp_path / "full" / "lstm"
        _write_summary(d, seed=42, git_commit=None, git_dirty=None)
        report = _common.check_provenance([d])
        assert report[0]["status"] == "missing"

    def test_multiple_dirs_mixed_status(self, tmp_path):
        clean_dir = tmp_path / "clean" / "lstm"
        dirty_dir = tmp_path / "dirty" / "lstm"
        missing_dir = tmp_path / "missing" / "lstm"
        _write_summary(clean_dir, seed=0, git_commit="aaa", git_dirty=False)
        _write_summary(dirty_dir, seed=1, git_commit="bbb", git_dirty=True)
        missing_dir.mkdir(parents=True)

        report = _common.check_provenance([clean_dir, dirty_dir, missing_dir])
        statuses = {r["dir"]: r["status"] for r in report}
        assert statuses[str(clean_dir)] == "clean"
        assert statuses[str(dirty_dir)] == "dirty"
        assert statuses[str(missing_dir)] == "missing"


class TestPrintProvenanceWarning:
    def test_silent_when_all_clean(self, capsys):
        report = [{"dir": "x", "status": "clean", "seed": 0, "git_commit": "a", "git_dirty": False}]
        _common.print_provenance_warning(report)
        assert capsys.readouterr().out == ""

    def test_warns_on_dirty(self, capsys):
        report = [
            {
                "dir": "results/full/lstm",
                "status": "dirty",
                "seed": 42,
                "git_commit": "d707aae7",
                "git_dirty": True,
            }
        ]
        _common.print_provenance_warning(report)
        out = capsys.readouterr().out
        assert "PROVENANCE WARNING" in out
        assert "DIRTY" in out
        assert "results/full/lstm" in out
        assert "d707aae7" in out

    def test_warns_on_missing(self, capsys):
        report = [
            {
                "dir": "results/full/lstm",
                "status": "missing",
                "seed": None,
                "git_commit": None,
                "git_dirty": None,
            }
        ]
        _common.print_provenance_warning(report)
        out = capsys.readouterr().out
        assert "MISSING" in out

    def test_empty_report_is_silent(self, capsys):
        _common.print_provenance_warning([])
        assert capsys.readouterr().out == ""


class TestMaskedMechanismM4c:
    """`build_masked_mechanism` gained Spring support so the
    graph-quality ladder can be run on a NON-dissipative family. The ladder is
    nearly flat on `full_nl` and the standing explanation is dissipation; that
    hypothesis is only testable if masking works on a family where effects
    persist. These guard the masking itself -- if it silently returned an
    unmasked mechanism the ladder would be flat for a trivial reason and the
    test of the hypothesis would be void.
    """

    @staticmethod
    def _graph():
        import numpy as np

        g = np.zeros((4, 4, 1))
        g[2, 1, 0] = 1.0
        g[3, 0, 0] = 1.0
        return g

    def test_spring_masking_changes_the_coupling(self):
        import numpy as np

        from causaltemp_xai.benchmarks.mechanisms import SpringMechanism

        g = self._graph()
        m = SpringMechanism(graph=g, k_spring=1.0, dt=0.1)
        masked = _common.build_masked_mechanism(m, np.zeros_like(g))
        assert masked.graph.sum() == 0, "masking must actually drop the edges"
        assert m.graph.sum() == 2, "the original must not be mutated"
        assert masked.k_spring == m.k_spring and masked.dt == m.dt

    def test_masking_actually_changes_the_rollout(self):
        """The real requirement: a masked mechanism must PREDICT differently.
        Equal parameters with an unused graph would pass the checks above and
        still make the ladder meaningless."""
        import numpy as np

        from causaltemp_xai.benchmarks.mechanisms import SpringMechanism

        g = self._graph()
        rng = np.random.default_rng(0)
        window = rng.normal(size=(1, 4))
        for m in (SpringMechanism(graph=g, k_spring=5.0, dt=0.1),):
            masked = _common.build_masked_mechanism(m, np.zeros_like(g))
            assert not np.allclose(
                m.forward_numpy(window), masked.forward_numpy(window)
            ), f"{type(m).__name__}: masked mechanism predicts identically"

    def test_unsupported_family_still_raises(self):
        import numpy as np
        import pytest

        class Bogus:
            graph = np.zeros((4, 4, 1))

        with pytest.raises(TypeError, match="requires an MLPMechanism"):
            _common.build_masked_mechanism(Bogus(), np.zeros((4, 4, 1)))
