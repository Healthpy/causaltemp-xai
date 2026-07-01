"""
Script 04: Evaluate all four metric axes on saved CF outputs.

Usage:
  python experiments/04_evaluate_axes.py --benchmark LinearSCMT --classifier lstm --method comte
  python experiments/04_evaluate_axes.py --benchmark LinearSCMT --classifier lstm --all_methods
"""

import argparse
import json
import pathlib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from causal_tscf_bench.benchmarks.base import BenchmarkDataset
from causal_tscf_bench.benchmarks.linear_scm_t import LinearSCMT
from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Nonmonotonic, NlinearSCMT_Regime
from causal_tscf_bench.classifiers.lstm import LSTMClassifier
from causal_tscf_bench.classifiers.transformer import TransformerClassifier
from causal_tscf_bench.metrics import compute_axis_b, compute_axis_c

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

BENCH_MAP = {
    "LinearSCMT": LinearSCMT,
    "NlinearSCMT": NlinearSCMT,
    "NlinearSCMT_Nonmonotonic": NlinearSCMT_Nonmonotonic,
    "NlinearSCMT_Regime": NlinearSCMT_Regime,
}
CLF_MAP = {"lstm": LSTMClassifier, "transformer": TransformerClassifier}


def evaluate(bench_name: str, clf_name: str, method_name: str) -> dict:
    split = BenchmarkDataset.load(str(DATA_DIR / bench_name))

    clf = CLF_MAP[clf_name]()
    clf.load(str(MODEL_DIR / bench_name / clf_name))

    res_dir = RESULTS_DIR / bench_name / clf_name
    cf_path = res_dir / f"X_cf_{method_name}.npy"
    if not cf_path.exists():
        print(f"     [SKIP] {cf_path} not found")
        return {}

    X_cf_exp = np.load(str(cf_path))
    N_cf = len(X_cf_exp)
    X_test = split.X_test[:N_cf]
    Y_test = split.Y_test[:N_cf]

    target_class = 1 - int(Y_test[0])

    # Ground-truth CFs: compute via causal ladder for each test instance
    print(f"     Computing {N_cf} ground-truth CFs ...")
    from causal_tscf_bench.scm.counterfactual import compute_gt_counterfactual
    X_cf_gt = np.zeros_like(X_test)
    causal_ch = int(split.meta.get("causal_channel", 0))
    T_int = X_test.shape[1] // 2

    for i in range(N_cf):
        # Intervention: negate the causal channel at T_int
        cur_val = float(X_test[i, T_int, causal_ch])
        int_val = -cur_val if abs(cur_val) > 0.1 else (1.0 if target_class == 1 else -1.0)
        try:
            X_cf_gt[i] = compute_gt_counterfactual(
                X_test[i], split.dag, split.mechanisms, causal_ch, T_int, int_val
            )
        except Exception:
            X_cf_gt[i] = X_test[i]  # fallback: factual

    # Axis B: graph quality (using ground-truth vs "no knowledge" baseline)
    adj_true = split.dag.adjacency.astype(int)
    adj_pred_zeros = np.zeros_like(adj_true)
    axis_b = compute_axis_b(adj_true, adj_pred_zeros, X=X_test)

    # Axis C: CF quality
    axis_c = compute_axis_c(
        X_test, X_cf_exp, X_cf_gt, split.X_train, clf, target_class, T_int
    )

    results = {
        "benchmark": bench_name,
        "classifier": clf_name,
        "method": method_name,
        "n_instances": N_cf,
        "axis_b": {k: (float(v) if isinstance(v, (int, float, np.floating)) and not np.isnan(v) else None)
                   for k, v in axis_b.items()},
        "axis_c": {k: (float(v) if isinstance(v, (int, float, np.floating)) and not np.isnan(v) else None)
                   for k, v in axis_c.items()},
    }

    out_path = res_dir / f"eval_{method_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"     Saved -> {out_path}")
    return results


def print_summary(results: dict) -> None:
    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    for section in ["axis_b", "axis_c"]:
        for k, v in results.get(section, {}).items():
            val_str = f"{v:.4f}" if v is not None else "N/A"
            print(f"  {k:<20} {val_str:>10}")


def _write_csv_tables(all_results, bench_name, clf_name):
    """Write results/tables/table1_axis_c.csv and table2_axis_b.csv."""
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    axis_c_metrics = ["Validity", "Proximity", "Sparsity", "OOD", "TRSI",
                       "CF_faith_mean", "CF_faith_std", "CF_faith_hard", "IVR"]
    axis_b_metrics = ["SHD", "LagAcc", "AUC", "TV_conf"]

    def _row(r, metrics, axis):
        vals = [f"{r[axis].get(m):.4f}" if r[axis].get(m) is not None else "NA"
                for m in metrics]
        return f"{r['benchmark']},{r['classifier']},{r['method']}," + ",".join(vals)

    # table1_axis_c.csv
    header_c = "benchmark,classifier,method," + ",".join(axis_c_metrics)
    rows_c = [header_c] + [_row(r, axis_c_metrics, "axis_c") for r in all_results]
    out_c = tables_dir / "table1_axis_c.csv"
    existing_c = out_c.read_text() if out_c.exists() else header_c
    # Append new rows (skip header if file exists)
    new_rows_c = rows_c[1:] if out_c.exists() else rows_c
    with open(out_c, "a") as f:
        f.write("\n".join(new_rows_c) + "\n")
    print(f"  [04] tables -> {out_c}")

    # table2_axis_b.csv
    header_b = "benchmark,classifier,method," + ",".join(axis_b_metrics)
    rows_b = [header_b] + [_row(r, axis_b_metrics, "axis_b") for r in all_results]
    out_b = tables_dir / "table2_axis_b.csv"
    new_rows_b = rows_b[1:] if out_b.exists() else rows_b
    with open(out_b, "a") as f:
        f.write("\n".join(new_rows_b) + "\n")
    print(f"  [04] tables -> {out_b}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCH_MAP))
    parser.add_argument("--classifier", required=True, choices=list(CLF_MAP))
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--all_methods", action="store_true")
    args = parser.parse_args()

    methods = ["wachter", "comte", "tsevo", "glacier", "cels", "confetti", "carla"] if args.all_methods else [args.method]

    all_results = []
    for method in methods:
        if method is None:
            continue
        print(f"\n[04] {args.benchmark} / {args.classifier} / {method}")
        r = evaluate(args.benchmark, args.classifier, method)
        if r:
            all_results.append(r)
            print_summary(r)

    # Write combined table
    if len(all_results) > 1:
        combined_path = RESULTS_DIR / args.benchmark / args.classifier / "all_methods_eval.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[04] Combined table -> {combined_path}")

        # Print comparison table
        key_metrics = ["Validity", "CF_faith_mean", "IVR", "TRSI", "Proximity"]
        print(f"\n{'Method':<12}", end="")
        for m in key_metrics:
            print(f"  {m:>14}", end="")
        print()
        print("-" * (12 + 16 * len(key_metrics)))
        for r in all_results:
            print(f"{r['method']:<12}", end="")
            for m in key_metrics:
                v = r["axis_c"].get(m)
                print(f"  {(f'{v:.4f}' if v is not None else 'N/A'):>14}", end="")
            print()

        # Write LaTeX-ready CSVs to results/tables/
        _write_csv_tables(all_results, args.benchmark, args.classifier)


if __name__ == "__main__":
    main()
