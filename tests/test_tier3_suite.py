"""Tests for `experiments/11_tier3_real_suite.py` (M4i,
2026-08-06).

`known`/`domain` modes are exercised deterministically here, against a small
synthetic `LinearMechanism` dumped through this file's own serialization
format -- no network, no real Tier-2/Tier-3 dataset needed, fully
reproducible. `discover` mode (which reuses Tier 2's real-data path) is
exercised only as a smoke run outside the pytest suite (M4i validation step),
since it needs an actual persisted real dataset and is slow.
"""

from __future__ import annotations

import importlib
import shutil

import numpy as np
import pytest

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism

_phase11 = importlib.import_module("experiments.11_tier3_real_suite")

_TEST_NAME = "__pytest_tier3_fixture__"


@pytest.fixture
def fixture_dataset(tmp_path):
    """A tiny fake persisted-CF directory under the real
    `results/real_<name>/lstm/cf/` path (matching `06b`'s/`07b`'s own
    unconditional-`ROOT` convention, not `--out-dir`-relative), so
    `run_known`/`run_domain` can score against it. Cleaned up after."""
    k, T, n = 4, 10, 5
    rng = np.random.default_rng(0)
    cf_dir = _phase11.ROOT / "results" / f"real_{_TEST_NAME}" / "lstm" / "cf"
    cf_dir.mkdir(parents=True, exist_ok=True)
    X_sel = rng.normal(size=(n, T, k))
    np.save(cf_dir / "X_sel.npy", X_sel)
    X_cf = X_sel.copy()
    X_cf[:, 5:, :] += rng.normal(scale=0.5, size=(n, T - 5, k))
    np.save(cf_dir / "X_cf_TestMethod.npy", X_cf)

    yield k, T

    tier3_root = _phase11.ROOT / "results" / f"real_{_TEST_NAME}"
    if tier3_root.exists():
        shutil.rmtree(tier3_root)


class TestSerialization:
    def test_mechanism_roundtrip(self, tmp_path):
        A0 = np.array([[0.0, 0.5], [0.3, 0.0]])
        A1 = np.array([[0.1, 0.0], [0.0, 0.2]])
        mech = LinearMechanism([A0, A1])
        path = tmp_path / "mech.npz"
        _phase11.save_mechanism_npz(mech, path)
        loaded = _phase11.load_mechanism_npz(path)
        assert loaded.L == 2
        assert loaded.k == 2
        assert np.array_equal(loaded.A_list[0], A0)
        assert np.array_equal(loaded.A_list[1], A1)

    def test_graph_roundtrip(self, tmp_path):
        graph = np.array([[[0], [1]], [[1], [0]]])
        path = tmp_path / "graph.npy"
        np.save(path, graph)
        loaded = _phase11.load_graph_npy(path)
        assert np.array_equal(loaded, graph)


class TestKnownMode:
    def test_graph_only_no_mechanism_notes_not_computable(self, fixture_dataset, tmp_path):
        k, _T = fixture_dataset
        graph = np.eye(k)[:, :, None].astype(int)
        graph_path = tmp_path / "graph.npy"
        np.save(graph_path, graph)

        _phase11.run_known(_TEST_NAME, None, graph_path, mechanism_file=None, skip_cf=True)

        out_path = (
            _phase11.ROOT
            / "results"
            / f"real_{_TEST_NAME}"
            / "tier3"
            / "known"
            / "graph_agreement.json"
        )
        assert out_path.exists()
        import json

        out = json.loads(out_path.read_text())
        assert "not computable" in out["cf_faith_note"]
        assert "cf_faith_by_method" not in out

    def test_with_mechanism_scores_persisted_cfs(self, fixture_dataset, tmp_path):
        k, _T = fixture_dataset
        rng = np.random.default_rng(1)
        A = rng.normal(scale=0.2, size=(k, k))
        np.fill_diagonal(A, 0)
        mech = LinearMechanism([A])
        mech_path = tmp_path / "mech.npz"
        _phase11.save_mechanism_npz(mech, mech_path)
        graph_path = tmp_path / "graph.npy"
        np.save(graph_path, (A != 0)[:, :, None].astype(int))

        _phase11.run_known(_TEST_NAME, None, graph_path, mechanism_file=mech_path, skip_cf=True)

        out_path = (
            _phase11.ROOT
            / "results"
            / f"real_{_TEST_NAME}"
            / "tier3"
            / "known"
            / "graph_agreement.json"
        )
        import json

        out = json.loads(out_path.read_text())
        assert "TestMethod" in out["cf_faith_by_method"]
        row = out["cf_faith_by_method"]["TestMethod"]
        assert "cf_faith_rollout" in row and "cf_faith_pearl" in row

    def test_missing_graph_file_raises(self):
        with pytest.raises(SystemExit):
            _phase11.dispatch("known", _TEST_NAME, None, graph_file=None)


class TestDomainMode:
    def test_graph_derived_from_mechanism_nonzero_entries(self, fixture_dataset, tmp_path):
        _k, _T = fixture_dataset
        A = np.array(
            [
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        mech = LinearMechanism([A])
        mech_path = tmp_path / "mech.npz"
        _phase11.save_mechanism_npz(mech, mech_path)

        _phase11.run_domain(_TEST_NAME, None, mech_path, skip_cf=True)

        out_path = (
            _phase11.ROOT
            / "results"
            / f"real_{_TEST_NAME}"
            / "tier3"
            / "domain"
            / "graph_agreement.json"
        )
        import json

        out = json.loads(out_path.read_text())
        assert out["graph_edges"] == int((A != 0).sum())

    def test_missing_mechanism_file_raises(self):
        with pytest.raises(SystemExit):
            _phase11.dispatch("domain", _TEST_NAME, None, mechanism_file=None)


class TestDispatchUnknownMode:
    def test_unknown_mode_raises(self):
        with pytest.raises(SystemExit):
            _phase11.dispatch("not_a_real_mode", _TEST_NAME, None)
