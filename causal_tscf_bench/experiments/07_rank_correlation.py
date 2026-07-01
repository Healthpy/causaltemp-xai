"""
Script 07: Spearman rank-correlation analysis (H4 extended).

Tests whether traditional CF metrics (Validity, Proximity, Sparsity) are
correlated with CF-faith across all (method × benchmark) pairs.

H4 claim: CF-faith is NOT captured by traditional metrics — traditional metrics
can be high even when CF-faith is low, exposing a measurement gap.

Produces:
  results/rank_correlation.json  — Spearman ρ and p-values for each metric pair
  results/tables/table3_rank_corr.csv  — formatted table

Usage:
  python experiments/07_rank_correlation.py --results_dir results/
"""

import argparse
import json
import pathlib
import numpy as np
from scipy import stats


TRADITIONAL = ["Validity", "Proximity", "Sparsity", "IVR", "TRSI"]
TARGET = "CF_faith_mean"


def load_eval_results(results_dir):
    records = []
    for fpath in results_dir.rglob("eval_*.json"):
        with open(fpath) as f:
            records.append(json.load(f))
    return records


def spearman_analysis(records):
    """
    Compute Spearman ρ between each traditional metric and CF_faith_mean.
    Returns dict of {metric: {"rho": float, "p_value": float, "n": int}}.
    """
    cf_faith_vals = [r["axis_c"].get(TARGET) for r in records
                     if r["axis_c"].get(TARGET) is not None]

    if len(cf_faith_vals) < 3:
        return {"error": "fewer than 3 records with CF_faith_mean"}

    results = {}
    for metric in TRADITIONAL:
        paired = [(r["axis_c"].get(metric), r["axis_c"].get(TARGET))
                  for r in records
                  if r["axis_c"].get(metric) is not None
                  and r["axis_c"].get(TARGET) is not None]
        if len(paired) < 3:
            results[metric] = {"skipped": True, "reason": f"n={len(paired)} < 3"}
            continue
        x_vals, y_vals = zip(*paired)
        rho, p = stats.spearmanr(x_vals, y_vals)
        results[metric] = {
            "rho": float(rho),
            "p_value": float(p),
            "n": len(paired),
            "significant": bool(p < 0.05),
        }
    return results


def per_method_analysis(records):
    """Spearman ρ stratified by method — reveals method-specific patterns."""
    methods = sorted(set(r["method"] for r in records))
    results = {}
    for method in methods:
        method_records = [r for r in records if r["method"] == method]
        if len(method_records) < 2:
            continue
        method_results = {}
        for metric in TRADITIONAL:
            paired = [(r["axis_c"].get(metric), r["axis_c"].get(TARGET))
                      for r in method_records
                      if r["axis_c"].get(metric) is not None
                      and r["axis_c"].get(TARGET) is not None]
            if len(paired) < 2:
                method_results[metric] = None
                continue
            if len(paired) < 3:
                x_vals, y_vals = zip(*paired)
                # Pearson as fallback for n=2
                try:
                    r_p, p_p = stats.pearsonr(x_vals, y_vals)
                    method_results[metric] = {"rho": float(r_p), "p_value": float(p_p),
                                              "n": 2, "test": "pearson"}
                except Exception:
                    method_results[metric] = None
            else:
                x_vals, y_vals = zip(*paired)
                rho, p = stats.spearmanr(x_vals, y_vals)
                method_results[metric] = {"rho": float(rho), "p_value": float(p),
                                          "n": len(paired), "test": "spearman"}
        results[method] = method_results
    return results


def write_csv(corr_results, tables_dir):
    """Write rank correlation results as a LaTeX-ready CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = ["Metric,Spearman_rho,p_value,n,significant"]
    for metric, res in corr_results.items():
        if isinstance(res, dict) and "rho" in res:
            rows.append(f"{metric},{res['rho']:.4f},{res['p_value']:.4f},"
                        f"{res['n']},{res['significant']}")
        elif isinstance(res, dict) and "skipped" in res:
            rows.append(f"{metric},NA,NA,0,False")
    out = tables_dir / "table3_rank_corr.csv"
    out.write_text("\n".join(rows))
    print(f"  [07] -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    records = load_eval_results(results_dir)
    print(f"[07] Loaded {len(records)} evaluation records")

    if not records:
        print("[07] No records found. Run 04_evaluate_axes.py first.")
        return

    print(f"[07] Methods present: {sorted(set(r['method'] for r in records))}")
    print(f"[07] Benchmarks present: {sorted(set(r.get('benchmark', '?') for r in records))}")

    overall = spearman_analysis(records)
    per_method = per_method_analysis(records)

    output = {
        "overall_spearman": overall,
        "per_method_spearman": per_method,
        "hypothesis": (
            "H4: If |rho| is low (< 0.3) and p > 0.05 for all traditional metrics, "
            "then CF-faith is not captured by traditional metrics."
        ),
    }

    out = results_dir / "rank_correlation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Overall Spearman rho: traditional metrics vs CF-faith ===")
    if "error" in overall:
        print(f"  {overall['error']}")
    else:
        print(f"  {'Metric':<12}  {'rho':>7}  {'p':>8}  {'n':>4}  significant")
        print(f"  {'-'*50}")
        for metric, res in overall.items():
            if isinstance(res, dict) and "rho" in res:
                print(f"  {metric:<12}  {res['rho']:>7.4f}  {res['p_value']:>8.4f}"
                      f"  {res['n']:>4}  {res['significant']}")
            elif isinstance(res, dict) and "skipped" in res:
                print(f"  {metric:<12}  {'NA':>7}  {'NA':>8}  {'0':>4}  (skipped: {res['reason']})")

    tables_dir = results_dir / "tables"
    write_csv(overall, tables_dir)

    print(f"\n[07] Saved -> {out}")


if __name__ == "__main__":
    main()
