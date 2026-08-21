"""Final-table schema and suppression-provenance regressions."""

from __future__ import annotations

import json
import sys

import pandas as pd

from experiments import make_final_table


def test_conditional_nan_is_suppressed_with_full_provenance():
    suppressed = []
    rows = make_final_table.rows_from_methods(
        "full",
        "lstm",
        [
            {
                "method": "NoiselessSCMRecourse",
                "validity": 0.0,
                "cf_faith_rollout_hard_valid": 0.0,
                "cf_faith_rollout_hard_given_valid": float("nan"),
            }
        ],
        suppressed,
    )

    emitted = {(row["method"], row["metric"]): row["value"] for row in rows}
    assert emitted[("NoiselessSCMRecourse", "cf_faith_rollout_hard_valid")] == 0.0
    assert ("NoiselessSCMRecourse", "cf_faith_rollout_hard_given_valid") not in emitted
    assert suppressed == [
        {
            "dataset": "full",
            "classifier": "lstm",
            "method": "NoiselessSCMRecourse",
            "metric": "cf_faith_rollout_hard_given_valid",
        }
    ]


def test_main_writes_suppressed_metric_keys_to_provenance(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    summary_dir = results_dir / "full" / "lstm"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "method": "PearlSCMRecourse",
                        "validity": 0.0,
                        "cf_faith_pearl_hard_valid": 0.0,
                        "cf_faith_pearl_hard_given_valid": float("nan"),
                    }
                ]
            }
        )
    )
    out = tmp_path / "tables" / "final_table_long.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_final_table.py",
            "--results-dir",
            str(results_dir),
            "--out",
            str(out),
            "--configs",
            "full",
        ],
    )

    make_final_table.main()

    provenance = json.loads((out.parent / "final_table_provenance.json").read_text())
    assert provenance["_suppressed_metrics"] == [
        {
            "dataset": "full",
            "classifier": "lstm",
            "method": "PearlSCMRecourse",
            "metric": "cf_faith_pearl_hard_given_valid",
        }
    ]


def test_explicit_publication_configs_exclude_retained_smoke(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    for dataset in ("full", "full_nl", "smoke", "smoke_nl"):
        summary_dir = results_dir / dataset / "lstm"
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps(
                {
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "git_worktree_dirty": True,
                    "methods": [{"method": "NoiselessSCMRecourse", "validity": 1.0}],
                }
            )
        )

    out = tmp_path / "tables" / "final_table_long.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_final_table.py",
            "--results-dir",
            str(results_dir),
            "--out",
            str(out),
            "--configs",
            "full",
            "full_nl",
        ],
    )
    make_final_table.main()

    long_df = pd.read_csv(out)
    assert set(long_df["dataset"]) == {"full", "full_nl"}
    provenance = json.loads((out.parent / "final_table_provenance.json").read_text())
    assert set(provenance) - {"_suppressed_metrics"} == {"full", "full_nl"}
    for dataset in ("full", "full_nl"):
        assert provenance[dataset]["lstm"]["git_dirty"] is False
        assert provenance[dataset]["lstm"]["git_worktree_dirty"] is True
