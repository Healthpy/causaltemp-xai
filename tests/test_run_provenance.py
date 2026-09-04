"""R7 gate: every result file carries a ``seed``, and (M2 DoD) a commit hash.

R7 ("every result file must include a ``seed`` field") was enforced only by
convention -- each phase hand-wrote ``seed`` into the ``provenance`` block of
``summary.json``. An audit on 2026-07-21 found **114 of 129** result JSONs had
no ``seed`` at all: ``eval_<Method>.json``, ``attribution.json``,
``axis_a_benchmark.json``, ``train_report.json`` and ``shift_vr.json`` were all
missing it, because a per-call-site convention is precisely what a new phase
forgets. (``attribution.json`` is named as it stood at that audit; the
attribution family was descoped 2026-07-29 and no phase writes that file now.
The graph diagnostic was ``axis_b_benchmark.json`` until the 2026-08-03
relettering.) Stamping moved into ``_common.dump_json`` so the gate holds for every
current and future phase; these tests are what keep it there.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import experiments._common as common
from experiments._common import (
    dump_json,
    git_provenance,
    run_provenance,
    set_run_context,
)


@pytest.fixture(autouse=True)
def _reset_context():
    """Each test declares its own context; don't leak between tests."""
    set_run_context(seed=-1, config="__unset__")
    yield
    set_run_context(seed=-1, config="__unset__")


class TestDumpJsonStampsProvenance:
    def test_seed_and_commit_land_in_a_plain_result_dict(self, tmp_path):
        """The regression that motivated this: a metrics dict written by a phase
        (e.g. eval_<Method>.json) must come back with seed + commit."""
        set_run_context(seed=7, config="smoke")
        out = tmp_path / "eval_NoiselessSCMRecourse.json"
        dump_json(out, {"validity": 1.0, "proximity_l1": 3.2})

        d = json.loads(out.read_text())
        assert d["seed"] == 7
        assert d["config"] == "smoke"
        assert "git_commit" in d
        assert d["validity"] == 1.0  # payload preserved

    def test_does_not_overwrite_a_payloads_own_keys(self, tmp_path):
        """A phase that already writes its own seed keeps it -- stamping only
        ever fills gaps, so it can never silently relabel a result."""
        set_run_context(seed=7, config="smoke")
        out = tmp_path / "summary.json"
        dump_json(out, {"seed": 99, "config": "explicit", "x": 1})

        d = json.loads(out.read_text())
        assert d["seed"] == 99
        assert d["config"] == "explicit"

    def test_non_dict_payload_is_written_unchanged(self, tmp_path):
        """Stamping must not corrupt a list payload into a dict."""
        out = tmp_path / "rows.json"
        dump_json(out, [{"a": 1}, {"a": 2}])
        assert json.loads(out.read_text()) == [{"a": 1}, {"a": 2}]

    def test_seed_survives_a_falsy_zero(self, tmp_path):
        """seed=0 is a real seed (smoke_seed0). A truthiness check would drop it."""
        set_run_context(seed=0, config="smoke_seed0")
        out = tmp_path / "r.json"
        dump_json(out, {"metric": 1.0})
        assert json.loads(out.read_text())["seed"] == 0


class TestGitProvenance:
    def test_reports_commit_and_dirty_flag(self):
        g = git_provenance()
        assert set(g) == {"git_commit", "git_dirty", "git_worktree_dirty"}
        if g["git_commit"] is not None:  # not a git checkout in some CI sandboxes
            assert len(g["git_commit"]) == 40
            assert isinstance(g["git_dirty"], bool)
            assert isinstance(g["git_worktree_dirty"], bool)

    def test_dirty_flag_is_present_alongside_the_hash(self):
        """A bare hash implies a cleanliness it may not have; the flag is what
        makes the hash honest, so the two must always travel together."""
        p = run_provenance()
        assert "git_commit" in p and "git_dirty" in p
        assert "git_worktree_dirty" in p

    def test_result_dirt_is_separate_from_source_dirt(self, tmp_path, monkeypatch):
        def git(*args):
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q")
        git("config", "user.email", "tests@example.invalid")
        git("config", "user.name", "Test Runner")
        (tmp_path / "source.py").write_text("VALUE = 1\n")
        results = tmp_path / "results"
        results.mkdir()
        (results / "artifact.json").write_text("{}\n")
        git("add", "source.py", "results/artifact.json")
        git("commit", "-qm", "fixture")

        monkeypatch.setattr(common, "ROOT", tmp_path)
        common.git_provenance.cache_clear()
        (results / "artifact.json").write_text('{"metric": 1}\n')
        result_only = common.git_provenance()
        assert result_only["git_dirty"] is False
        assert result_only["git_worktree_dirty"] is True

        common.git_provenance.cache_clear()
        (tmp_path / "source.py").write_text("VALUE = 2\n")
        source_changed = common.git_provenance()
        assert source_changed["git_dirty"] is True
        assert source_changed["git_worktree_dirty"] is True
        common.git_provenance.cache_clear()
