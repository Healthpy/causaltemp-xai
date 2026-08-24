"""Regression coverage for the classifier-scored structural oracle phase."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np

from causaltemp_xai.benchmarks.generator import LinearSCMT

_phase05 = importlib.import_module("experiments.05_run_oracle_control")


def test_oracle_phase_loads_checkpoint_and_emits_outcome_metrics(tmp_path, monkeypatch):
    generated = LinearSCMT(k=3, L=1, T=12, N=20, seed=3).generate()
    data = {
        **generated,
        "X_train": generated["X"][:10],
        "Y_train": generated["Y"][:10],
        "X_test": generated["X"][10:],
        "Y_test": generated["Y"][10:],
    }
    cfg = SimpleNamespace(
        name="oracle_fixture",
        seed=3,
        mechanism_type="linear",
        noise_type="laplace",
        k=3,
        T=12,
        as_dict=lambda: {"name": "oracle_fixture", "seed": 3},
    )
    out_dir = tmp_path / "data"
    checkpoint = out_dir / cfg.name / "lstm.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    result_dir = tmp_path / "results" / cfg.name / "oracle"
    loaded_paths = []

    class _Classifier:
        def predict(self, Z):
            return np.ones(len(np.asarray(Z)), dtype=int)

    monkeypatch.setattr(_phase05, "get_config", lambda name: cfg)
    monkeypatch.setattr(_phase05, "load_dataset", lambda name, out_dir: data)
    monkeypatch.setattr(
        _phase05,
        "LSTMClassifier",
        SimpleNamespace(load=lambda path: (loaded_paths.append(path), _Classifier())[1]),
    )
    monkeypatch.setattr(_phase05, "config_dir", lambda name, classifier: result_dir)
    monkeypatch.setattr(_phase05, "TABLES_DIR", tmp_path / "results" / "tables")

    _phase05.run(cfg.name, n_cf=3, out_dir=out_dir)

    assert loaded_paths == [checkpoint]
    summary = json.loads((result_dir / "summary.json").read_text())
    assert isinstance(summary["methods"], list)
    assert {row["method"] for row in summary["methods"]} == {
        "OracleCF-Pearl",
        "OracleCF-Rollout",
    }
    required = {
        "validity",
        "ood",
        "do_complexity_mean_all",
        "do_complexity_mean_pearl_scorable",
        "n_do_scorable",
        "frac_no_do_schedule",
    }
    for row in summary["methods"]:
        assert required <= row.keys()
    pearl = next(row for row in summary["methods"] if row["method"] == "OracleCF-Pearl")
    assert pearl["do_complexity_mean_pearl_scorable"] == 1.0
    assert pearl["n_do_scorable"] == 3
