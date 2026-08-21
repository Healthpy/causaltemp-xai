#!/usr/bin/env python3
"""Produce the FINAL long-format results table for causaltemp-xai.

Walks `results/<config>/<classifier>/` directories and emits one long-format
CSV with columns (dataset, classifier, method, metric, value), plus a wide
pivot for humans and a provenance JSON.

Dependencies: stdlib + pandas only. No torch, no causaltemp_xai imports --
this must run anywhere, including a login node or a laptop.

Usage:
    python experiments/make_final_table.py \
        [--results-dir results] \
        [--out results/tables/final_table_long.csv] \
        [--configs smoke full ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Metric name lists (verified against real smoke summary.json / pns.json on
# 2026-08-17; see the report for the discrepancies vs. the original spec).
# ---------------------------------------------------------------------------

METHOD_METRICS = [
    "validity",
    "proximity_l1",
    "proximity_l2",
    "sparsity",
    "sparsity_channels",
    "sparsity_timepoints",
    "trsi",
    "scm_noise_plausibility",
    "ood",
    "cf_faith_rollout_hard",
    "cf_faith_rollout_soft",
    "cf_faith_pearl_hard",
    "cf_faith_pearl_soft",
    "cf_faith_rollout_hard_valid",
    "cf_faith_pearl_hard_valid",
    "cf_faith_rollout_hard_given_valid",
    "cf_faith_pearl_hard_given_valid",
    "frac_vacuous",
    "frac_degenerate",
    "do_complexity_mean",
    "do_complexity_mean_all",
    "do_complexity_mean_pearl_scorable",
    "do_complexity_median",
    "do_complexity_stability",
    "n_do_scorable",
    "frac_no_do_schedule",
    "n",
    "n_cf_faith_scorable",
]

SHIFT_VR_METRICS = ["shift_vr", "validity_base", "validity_shift"]

# Metadata keys that live alongside the real metrics in axis_a / shift_vr /
# pns.json top-level dicts and must NOT be mistaken for a method name or a
# metric.
SUMMARY_METADATA_KEYS = {"seed", "config", "git_commit", "git_dirty", "note"}
PNS_METADATA_KEYS = {
    "git_commit",
    "git_dirty",
    "seed",
    "config",
    "label_threshold",
    "label_fn",
    "label_site",
    "T",
    "schedule_mode",
    "direction",
    "PN_world",
    "PN_note",
}

PER_INSTANCE_FALLBACK_METRICS = [
    "validity",
    "proximity_l1",
    "proximity_l2",
    "sparsity",
    "sparsity_channels",
    "sparsity_timepoints",
    "trsi",
    "scm_noise_plausibility",
    "cf_faith_rollout_hard",
    "cf_faith_rollout_soft",
    "cf_faith_pearl_hard",
    "cf_faith_pearl_soft",
    "cf_faith_rollout_hard_valid",
    "cf_faith_pearl_hard_valid",
    "cf_faith_rollout_hard_given_valid",
    "cf_faith_pearl_hard_given_valid",
]

HEADLINE_COLUMNS = [
    "validity",
    "proximity_l1",
    "sparsity",
    "cf_faith_rollout_hard",
    "cf_faith_pearl_hard",
    "do_complexity_mean",
    "do_complexity_mean_all",
    "do_complexity_mean_pearl_scorable",
]


def is_metric_value(v) -> bool:
    """True for values we should emit as a metric: real numbers, not bool/NaN/None."""
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        try:
            if pd.isna(v):
                return False
        except (TypeError, ValueError):
            pass
        return True
    return False


def discover_configs(results_dir: Path) -> list[str]:
    configs = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(d.glob("*/summary.json")):
            configs.append(d.name)
    return configs


def rows_from_methods(
    dataset: str,
    classifier: str,
    methods: list[dict],
    suppressed_metrics: list[dict] | None = None,
) -> list[dict]:
    rows = []
    for m in methods:
        method = m.get("method")
        if not method:
            continue
        for metric in METHOD_METRICS:
            if metric not in m:
                continue
            v = m[metric]
            if not is_metric_value(v):
                if (
                    suppressed_metrics is not None
                    and metric.endswith("_hard_given_valid")
                    and v is not None
                    and pd.isna(v)
                ):
                    suppressed_metrics.append(
                        {
                            "dataset": dataset,
                            "classifier": classifier,
                            "method": method,
                            "metric": metric,
                        }
                    )
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "method": method,
                    "metric": metric,
                    "value": float(v),
                }
            )
    return rows


def rows_from_shift_vr(dataset: str, classifier: str, shift_vr: dict) -> list[dict]:
    rows = []
    for method, d in shift_vr.items():
        if method in SUMMARY_METADATA_KEYS:
            continue
        if not isinstance(d, dict):
            continue
        for metric in SHIFT_VR_METRICS:
            if metric not in d:
                continue
            v = d[metric]
            if not is_metric_value(v):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "method": method,
                    "metric": metric,
                    "value": float(v),
                }
            )
    return rows


def rows_from_axis_a(dataset: str, classifier: str, axis_a: dict) -> list[dict]:
    rows = []
    for key, v in axis_a.items():
        if key in SUMMARY_METADATA_KEYS:
            continue
        if not is_metric_value(v):
            continue
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "method": "__dataset__",
                "metric": f"axis_a_{key}",
                "value": float(v),
            }
        )
    return rows


def rows_from_pns(dataset: str, classifier: str, pns: dict) -> list[dict]:
    rows = []
    for direction in ("PS", "PN"):
        direction_dict = pns.get(direction)
        if not isinstance(direction_dict, dict):
            continue
        for method, metrics in direction_dict.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, v in metrics.items():
                if metric_name in PNS_METADATA_KEYS:
                    continue
                if not is_metric_value(v):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "method": method,
                        "metric": f"pns_{direction}_{metric_name}",
                        "value": float(v),
                    }
                )
    return rows


def rows_from_per_instance_fallback(dataset: str, classifier: str, csv_path: Path) -> list[dict]:
    rows = []
    df = pd.read_csv(csv_path)
    if "method" not in df.columns:
        return rows
    for method, g in df.groupby("method"):
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "method": method,
                "metric": "n",
                "value": float(len(g)),
            }
        )
        for metric in PER_INSTANCE_FALLBACK_METRICS:
            if metric not in g.columns:
                continue
            mean_v = g[metric].mean(skipna=True)
            if not is_metric_value(mean_v):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "method": method,
                    "metric": metric,
                    "value": float(mean_v),
                }
            )
        if "do_complexity" in g.columns:
            do_values = pd.to_numeric(g["do_complexity"], errors="coerce").dropna()
            do_scorable = do_values[do_values > 0]
            explicit_do_metrics = {
                "do_complexity_mean": do_values.mean(),
                "do_complexity_mean_all": do_values.mean(),
                "do_complexity_mean_pearl_scorable": do_scorable.mean(),
                "n_do_scorable": len(do_scorable),
                "frac_no_do_schedule": 1.0 - len(do_scorable) / len(g),
            }
            for metric, value in explicit_do_metrics.items():
                if not is_metric_value(value):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "method": method,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def process_config(results_dir: Path, dataset: str) -> tuple[list[dict], dict, list[dict]]:
    """Return rows, dataset provenance, and explicitly suppressed metric keys."""
    config_dir = results_dir / dataset
    rows: list[dict] = []
    provenance: dict = {}
    suppressed_metrics: list[dict] = []

    for classifier_dir in sorted(config_dir.iterdir()):
        if not classifier_dir.is_dir():
            continue
        classifier = classifier_dir.name
        summary_path = classifier_dir / "summary.json"

        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            methods = summary.get("methods", [])
            rows.extend(rows_from_methods(dataset, classifier, methods, suppressed_metrics))

            if "shift_vr" in summary and isinstance(summary["shift_vr"], dict):
                rows.extend(rows_from_shift_vr(dataset, classifier, summary["shift_vr"]))

            if "axis_a" in summary and isinstance(summary["axis_a"], dict):
                rows.extend(rows_from_axis_a(dataset, classifier, summary["axis_a"]))

            provenance[classifier] = {
                "seed": summary.get("seed"),
                "git_commit": summary.get("git_commit"),
                "git_dirty": summary.get("git_dirty"),
                "config": summary.get("provenance", {}).get("config"),
            }
        else:
            per_instance_path = classifier_dir / "per_instance.csv"
            if per_instance_path.exists():
                rows.extend(rows_from_per_instance_fallback(dataset, classifier, per_instance_path))
                provenance[classifier] = {
                    "note": "summary.json missing; metrics derived from per_instance.csv fallback"
                }
            else:
                continue

        pns_path = classifier_dir / "pns.json"
        if pns_path.exists():
            with open(pns_path) as f:
                pns = json.load(f)
            rows.extend(rows_from_pns(dataset, classifier, pns))

    return rows, provenance, suppressed_metrics


def build_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    dedup = long_df.drop_duplicates(
        subset=["dataset", "classifier", "method", "metric"], keep="last"
    )
    wide = dedup.pivot_table(
        index=["dataset", "classifier", "method"],
        columns="metric",
        values="value",
        aggfunc="last",
    )
    wide = wide.reset_index()
    return wide


def print_markdown_preview(wide_df: pd.DataFrame) -> None:
    if wide_df.empty:
        print("(no data)")
        return
    cols = ["dataset", "classifier", "method"] + [
        c for c in HEADLINE_COLUMNS if c in wide_df.columns
    ]
    preview = wide_df[cols].copy()
    for c in HEADLINE_COLUMNS:
        if c in preview.columns:
            preview[c] = preview[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in preview.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/tables/final_table_long.csv")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Restrict to these config/dataset names; default is auto-discover.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.configs:
        configs = args.configs
    else:
        configs = discover_configs(results_dir)

    if not configs:
        print(f"No configs with */summary.json found under {results_dir}")

    all_rows: list[dict] = []
    all_provenance: dict = {}
    all_suppressed_metrics: list[dict] = []
    for dataset in configs:
        rows, provenance, suppressed_metrics = process_config(results_dir, dataset)
        all_rows.extend(rows)
        all_provenance[dataset] = provenance
        all_suppressed_metrics.extend(suppressed_metrics)

    long_df = pd.DataFrame(all_rows, columns=["dataset", "classifier", "method", "metric", "value"])
    long_df = long_df.drop_duplicates(
        subset=["dataset", "classifier", "method", "metric"], keep="last"
    )
    long_df = long_df.sort_values(["dataset", "classifier", "method", "metric"]).reset_index(
        drop=True
    )
    long_df.to_csv(out_path, index=False)
    print(f"Wrote long-format table: {out_path} ({len(long_df)} rows)")

    wide_df = build_wide(long_df)
    wide_path = out_path.parent / "final_table_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"Wrote wide-format table: {wide_path} ({len(wide_df)} rows)")

    provenance_path = out_path.parent / "final_table_provenance.json"
    all_provenance["_suppressed_metrics"] = all_suppressed_metrics
    with open(provenance_path, "w") as f:
        json.dump(all_provenance, f, indent=2, default=str)
    print(f"Wrote provenance: {provenance_path}")

    print()
    print(f"Configs included: {configs}")
    print()
    print_markdown_preview(wide_df)


if __name__ == "__main__":
    main()
