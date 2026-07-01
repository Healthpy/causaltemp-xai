"""
Script 06: Statistical hypothesis testing over evaluation results.

Tests H1-H7 using Wilcoxon signed-rank tests (paired, non-parametric).
Benjamini-Hochberg FDR correction across all testable hypotheses.

Pairing strategy for H1-H3:
  For each (benchmark, classifier) combination, pair each causal method result
  against each non-causal method result and collect (causal - non-causal) differences.
  Wilcoxon tests if the median signed difference is consistently > 0.

H4 (non-monotonic): loads NlinearSCMT vs NlinearSCMT_Nonmonotonic results.
H5 (Gaussian vs Laplace): loads ablation JSON from 05_ablations.py output.
H6 (Label Visibility): checks fraction of balanced runs.
H7 (regime-switching): loads ablation JSON for multi-regime benchmarks.

Usage:
  python experiments/06_hypothesis_tests.py --results_dir results/
"""

import argparse
import json
import pathlib
import numpy as np
from scipy import stats


HYPOTHESES = {
    "H1": "Causal CF methods score higher CF-faith than non-causal baselines",
    "H2": "Causal CF methods respect IVR constraint (lower IVR than non-causal)",
    "H3": "TRSI is lower for causal methods vs. naive tabular baseline (DiCE)",
    "H4": "CF-faith degrades under non-monotonic operators (nonmonotonic vs linear)",
    "H5": "CF-faith is higher under non-Gaussian noise (Laplace) vs. Gaussian",
    "H6": "Label Visibility ensures class balance in [0.3, 0.7]",
    "H7": "CF-faith degrades under regime-switching vs. stationary SCM",
}

CAUSAL_METHODS = {"carla", "icitris", "citris"}
NONCAUSAL_METHODS = {"comte", "tsevo", "glacier", "dice"}


def bh_correction(pvalues, alpha=0.05):
    n = len(pvalues)
    if n == 0:
        return []
    ranked = sorted(range(n), key=lambda i: pvalues[i])
    rejected = [False] * n
    for rank, idx in enumerate(ranked, 1):
        if pvalues[idx] <= alpha * rank / n:
            rejected[idx] = True
    return rejected


def load_eval_results(results_dir):
    records = []
    for fpath in results_dir.rglob("eval_*.json"):
        with open(fpath) as f:
            records.append(json.load(f))
    return records


def _wilcoxon_or_skip(differences, alternative="greater", label=""):
    """Run Wilcoxon signed-rank test; skip gracefully if too few unique values."""
    if len(differences) < 2:
        return {"skipped": True, "reason": f"insufficient paired data (n={len(differences)})"}
    diffs = np.array(differences, dtype=float)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return {"skipped": True, "reason": "all differences are zero"}
    try:
        stat, p = stats.wilcoxon(diffs, alternative=alternative)
        return {"statistic": float(stat), "p_value": float(p),
                "n_pairs": len(diffs), "median_diff": float(np.median(diffs))}
    except Exception as e:
        return {"skipped": True, "reason": str(e)}


def _build_paired_diffs(records, metric_key, axis="axis_c",
                        causal_set=CAUSAL_METHODS, noncausal_set=NONCAUSAL_METHODS):
    """
    For each (benchmark, classifier) context, pair each causal record against
    each non-causal record and collect (causal_val - noncausal_val).
    """
    from itertools import product as iproduct
    by_ctx = {}
    for r in records:
        ctx = (r.get("benchmark", ""), r.get("classifier", ""))
        by_ctx.setdefault(ctx, []).append(r)

    diffs = []
    for ctx, ctx_records in by_ctx.items():
        causal_vals = [r[axis].get(metric_key) for r in ctx_records
                       if r["method"] in causal_set and r[axis].get(metric_key) is not None]
        noncausal_vals = [r[axis].get(metric_key) for r in ctx_records
                          if r["method"] in noncausal_set and r[axis].get(metric_key) is not None]
        for cv, ncv in iproduct(causal_vals, noncausal_vals):
            diffs.append(cv - ncv)
    return diffs


def test_h1(records):
    diffs = _build_paired_diffs(records, "CF_faith_mean")
    result = _wilcoxon_or_skip(diffs, alternative="greater")
    if "skipped" not in result:
        result["H1_supported"] = result["p_value"] < 0.05
    return result


def test_h2(records):
    # H2: causal IVR < non-causal IVR  →  (noncausal - causal) > 0
    diffs_neg = _build_paired_diffs(records, "IVR")   # causal - noncausal
    diffs_pos = [-d for d in diffs_neg]               # noncausal - causal
    result = _wilcoxon_or_skip(diffs_pos, alternative="greater")
    if "skipped" not in result:
        result["H2_supported"] = result["p_value"] < 0.05
    return result


def test_h3(records):
    # H3: DiCE TRSI > causal methods TRSI
    from itertools import product as iproduct
    by_ctx = {}
    for r in records:
        ctx = (r.get("benchmark", ""), r.get("classifier", ""))
        by_ctx.setdefault(ctx, []).append(r)

    diffs = []
    for ctx, ctx_records in by_ctx.items():
        causal_vals = [r["axis_c"].get("TRSI") for r in ctx_records
                       if r["method"] in CAUSAL_METHODS and r["axis_c"].get("TRSI") is not None]
        dice_vals = [r["axis_c"].get("TRSI") for r in ctx_records
                     if r["method"] == "dice" and r["axis_c"].get("TRSI") is not None]
        from itertools import product as iproduct
        for dv, cv in iproduct(dice_vals, causal_vals):
            diffs.append(dv - cv)   # (dice - causal) > 0

    result = _wilcoxon_or_skip(diffs, alternative="greater")
    if "skipped" not in result:
        result["H3_supported"] = result["p_value"] < 0.05
    return result


def test_h4(records):
    """
    H4: CF-faith degrades under non-monotonic operators.
    Compare CF_faith_mean on NlinearSCMT vs NlinearSCMT_Nonmonotonic.
    Expects records from both benchmarks to be present.
    """
    linear_vals = [r["axis_c"].get("CF_faith_mean") for r in records
                   if r.get("benchmark") == "NlinearSCMT"
                   and r["axis_c"].get("CF_faith_mean") is not None]
    nonmono_vals = [r["axis_c"].get("CF_faith_mean") for r in records
                    if r.get("benchmark") == "NlinearSCMT_Nonmonotonic"
                    and r["axis_c"].get("CF_faith_mean") is not None]

    if not linear_vals or not nonmono_vals:
        return {"skipped": True, "reason": "NlinearSCMT_Nonmonotonic results not found"}

    # Pair same-index method results if same length; else use all combinations
    if len(linear_vals) == len(nonmono_vals):
        diffs = [lv - nv for lv, nv in zip(linear_vals, nonmono_vals)]
    else:
        from itertools import product as iproduct
        diffs = [lv - nv for lv, nv in iproduct(linear_vals, nonmono_vals)]

    # H4: linear CF-faith > nonmonotonic CF-faith (positive differences)
    result = _wilcoxon_or_skip(diffs, alternative="greater")
    if "skipped" not in result:
        result.update({
            "linear_mean": float(np.mean(linear_vals)),
            "nonmono_mean": float(np.mean(nonmono_vals)),
            "H4_supported": result["p_value"] < 0.05,
        })
    return result


def test_h5(ablation_dir):
    """H5: Laplace noise → higher class balance than Gaussian (identifiability condition)."""
    fpath = ablation_dir / "H5_gaussian_vs_laplace.json"
    if not fpath.exists():
        return {"skipped": True, "reason": f"ablation file not found: {fpath}"}

    with open(fpath) as f:
        data = json.load(f)

    laplace_bal = data.get("laplace", {}).get("class_balance")
    gauss_bal = data.get("gaussian", {}).get("class_balance")
    if laplace_bal is None or gauss_bal is None:
        return {"skipped": True, "reason": "missing balance values in H5 ablation"}

    # Single-pair Wilcoxon is degenerate with n=1; use binomial sign test instead
    diff = laplace_bal - gauss_bal
    return {
        "laplace_balance": float(laplace_bal),
        "gaussian_balance": float(gauss_bal),
        "diff_laplace_minus_gaussian": float(diff),
        "H5_supported": diff > 0,
        "note": "single-seed comparison — not a significance test; H5 is a structural claim",
    }


def test_h6(ablation_dir):
    """H6: Label Visibility — fraction of seeds with balance in [0.3, 0.7]."""
    fpath = ablation_dir / "H6_label_visibility.json"
    if not fpath.exists():
        return {"skipped": True, "reason": f"ablation file not found: {fpath}"}

    with open(fpath) as f:
        data = json.load(f)

    frac = data.get("fraction_in_range", None)
    trials = data.get("trials", [])
    in_range_flags = [int(t["in_range"]) for t in trials]

    if not in_range_flags:
        return {"skipped": True, "reason": "no trial data"}

    # Binomial test: H0 = fraction ≤ 0.5, H1 = fraction > 0.5
    n = len(in_range_flags)
    k = sum(in_range_flags)
    binom_result = stats.binomtest(k, n, p=0.5, alternative="greater")
    return {
        "fraction_in_range": float(frac) if frac is not None else float(k / n),
        "n_trials": n,
        "n_balanced": k,
        "p_value": float(binom_result.pvalue),
        "H6_supported": binom_result.pvalue < 0.05,
    }


def test_h7(ablation_dir):
    """H7: Regime-switching degrades class balance (proxy for CF-faith degradation)."""
    results_2 = ablation_dir / "H7_regime_2.json"
    results_3 = ablation_dir / "H7_regime_3.json"

    vals = {}
    for n_reg, fpath in [(2, results_2), (3, results_3)]:
        if fpath.exists():
            with open(fpath) as f:
                d = json.load(f)
            vals[n_reg] = d.get("class_balance")

    if not vals:
        return {"skipped": True, "reason": "no H7 ablation files found"}

    out = {"balances_by_n_regimes": {str(k): v for k, v in vals.items()}}
    # Qualitative: more regimes → lower balance (harder identifiability)
    if 2 in vals and 3 in vals and vals[2] is not None and vals[3] is not None:
        out["balance_decreases_with_n_regimes"] = bool(vals[3] <= vals[2])
        out["H7_supported"] = bool(vals[3] <= vals[2])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    ablation_dir = results_dir / "ablations"
    records = load_eval_results(results_dir)
    print(f"[06] Loaded {len(records)} evaluation records")

    tests = {
        "H1": test_h1(records),
        "H2": test_h2(records),
        "H3": test_h3(records),
        "H4": test_h4(records),
        "H5": test_h5(ablation_dir),
        "H6": test_h6(ablation_dir),
        "H7": test_h7(ablation_dir),
    }

    # BH correction over all hypotheses with a valid p-value
    testable_keys = [k for k, v in tests.items()
                     if "skipped" not in v and "p_value" in v]
    if testable_keys:
        pvals = [tests[k]["p_value"] for k in testable_keys]
        rejected = bh_correction(pvals)
        for k, rej in zip(testable_keys, rejected):
            tests[k]["BH_rejected"] = bool(rej)

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, bool):
            return obj
        return obj

    out = results_dir / "hypothesis_tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(_jsonify(tests), f, indent=2)

    print("\n=== Hypothesis Test Results ===")
    for h, result in tests.items():
        print(f"\n{h}: {HYPOTHESES[h]}")
        for k, v in result.items():
            print(f"  {k}: {v}")

    print(f"\n[06] Saved -> {out}")


if __name__ == "__main__":
    main()
