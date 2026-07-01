"""
Script 05b: Evaluate Axis A — Concept Quality — across attribution and concept methods.

Methods evaluated:
  timeshap   TimeSHAP attributions (attr_timeshap.npy) — loaded or generated
  dynamask   Dynamask attributions (attr_dynamask.npy) — loaded or generated
  cbmt       CBM-T linear concept probes (trained here)
  ivae       iVAE disentanglement (trained here; provides LD + MCC_disent)

Axis A metrics (from causal_tscf_bench.metrics.axis_a):
  ICC          Fraction of attribution mass on ground-truth intervened channel
  MCC_coverage Fraction of causal parent channels with non-zero attribution
  LD           Linear R² alignment between iVAE latent dims and channel means
  MCC_disent   Hyvärinen MCC between iVAE latents and ground-truth factors

Usage:
  python experiments/05_evaluate_axis_a.py --benchmark LinearSCMT --classifier lstm
  python experiments/05_evaluate_axis_a.py --benchmark NlinearSCMT --classifier lstm --all_benchmarks
"""

import argparse
import json
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from causal_tscf_bench.benchmarks.base import BenchmarkDataset
from causal_tscf_bench.benchmarks.linear_scm_t import LinearSCMT
from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Nonmonotonic, NlinearSCMT_Regime
from causal_tscf_bench.classifiers.lstm import LSTMClassifier
from causal_tscf_bench.classifiers.transformer import TransformerClassifier
from causal_tscf_bench.metrics import compute_axis_a
from causal_tscf_bench.methods.concept.cbm_t import CBMT
from causal_tscf_bench.methods.concept.ivae import iVAE

DATA_DIR  = pathlib.Path(__file__).parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

BENCH_MAP = {
    "LinearSCMT": LinearSCMT,
    "NlinearSCMT": NlinearSCMT,
    "NlinearSCMT_Nonmonotonic": NlinearSCMT_Nonmonotonic,
    "NlinearSCMT_Regime": NlinearSCMT_Regime,
}
CLF_MAP = {"lstm": LSTMClassifier, "transformer": TransformerClassifier}


def _get_attr(method_name: str, split, classifier, res_dir: pathlib.Path) -> np.ndarray:
    """Load existing attribution file or run the method and save it."""
    attr_path = res_dir / f"attr_{method_name}.npy"
    if attr_path.exists():
        print(f"     [load] {attr_path.name}")
        return np.load(str(attr_path))

    print(f"     [run]  computing {method_name} attributions ...")
    if method_name == "timeshap":
        from causal_tscf_bench.methods.attribution.timeshap import TimeSHAP
        method = TimeSHAP()
    elif method_name == "dynamask":
        from causal_tscf_bench.methods.attribution.dynamask import Dynamask
        method = Dynamask()
    else:
        raise ValueError(f"Unknown attribution method: {method_name}")

    if hasattr(method, "fit"):
        method.fit(split.X_train)

    N = len(split.X_test)
    attrs = np.zeros_like(split.X_test)
    for i in range(N):
        attrs[i] = method.attribute(split.X_test[i], classifier)
        if (i + 1) % 50 == 0:
            print(f"       [{i+1}/{N}]")
    np.save(str(attr_path), attrs)
    print(f"     Saved -> {attr_path}")
    return attrs


def _causal_parents_list(dag, causal_channel: int, N: int):
    """Return list-of-lists: causal parents of causal_channel for each of N instances."""
    # parents_of(j) returns list of (i, lag) pairs — extract unique channel indices
    parents = [i for i, _lag in dag.parents_of(causal_channel)]
    return [parents] * N


def evaluate_axis_a(bench_name: str, clf_name: str) -> dict:
    split = BenchmarkDataset.load(str(DATA_DIR / bench_name))
    clf   = CLF_MAP[clf_name]()
    clf.load(str(MODEL_DIR / bench_name / clf_name))

    res_dir = RESULTS_DIR / bench_name / clf_name
    res_dir.mkdir(parents=True, exist_ok=True)

    N, T, M = split.X_test.shape
    causal_ch = int(split.meta.get("causal_channel", 0))
    parents_list = _causal_parents_list(split.dag, causal_ch, N)
    int_channels  = np.full(N, causal_ch, dtype=int)

    # Channel means for LD (ground-truth factor proxy)
    X_channels = split.X_test.mean(axis=1)  # (N, M)

    all_results = {}

    # ── 1. Attribution-based methods (TimeSHAP, Dynamask, CBM-T) ────────────
    for attr_name in ["timeshap", "dynamask"]:
        print(f"\n  [05a] {bench_name}/{clf_name} — {attr_name}")
        attrs = _get_attr(attr_name, split, clf, res_dir)
        metrics = compute_axis_a(attrs, int_channels, parents_list)
        all_results[attr_name] = metrics
        _print_metrics(attr_name, metrics)

    # ── 2. CBM-T ─────────────────────────────────────────────────────────────
    print(f"\n  [05a] {bench_name}/{clf_name} — cbmt")
    cbmt = CBMT()
    # Concept labels: binary activation — channel above its median
    concept_labels = (split.X_train.mean(axis=1) >
                      np.median(split.X_train.mean(axis=1), axis=0)).astype(int)  # (N, M)
    cbmt.fit_concepts(split.X_train, concept_labels)

    attrs_cbmt = np.zeros((N, T, M), dtype=np.float64)
    for i in range(N):
        attrs_cbmt[i] = cbmt.attribute(split.X_test[i], clf)
    np.save(str(res_dir / "attr_cbmt.npy"), attrs_cbmt)

    metrics_cbmt = compute_axis_a(attrs_cbmt, int_channels, parents_list)
    all_results["cbmt"] = metrics_cbmt
    _print_metrics("cbmt", metrics_cbmt)

    # ── 3. iVAE — latent disentanglement ────────────────────────────────────
    print(f"\n  [05a] {bench_name}/{clf_name} — ivae (training ...)")
    ivae = iVAE(latent_dim=M, n_segments=4, epochs=30, batch_size=64)
    ivae.fit_unsupervised(split.X_train)

    Z_test  = ivae.encode(split.X_test)   # (N, M)
    Z_train = ivae.encode(split.X_train)  # (N_train, M)

    # Use training channel means as Z_true proxy (iVAE has no ground-truth z)
    # LD is computed on test set; MCC_disent needs paired Z_true — use channel
    # activations as a stand-in (same shape as Z)
    X_ch_train = split.X_train.mean(axis=1)  # (N_train, M)

    metrics_ivae = compute_axis_a(
        np.zeros((N, T, M)),        # attributions not used for iVAE
        int_channels, parents_list,
        Z=Z_test,
        X_channels=X_channels,
    )
    all_results["ivae"] = metrics_ivae
    _print_metrics("ivae", metrics_ivae)

    # ── Save combined ────────────────────────────────────────────────────────
    out = {
        "benchmark": bench_name,
        "classifier": clf_name,
        "n_instances": N,
        "causal_channel": causal_ch,
        "causal_parents": parents_list[0],
        "methods": {
            k: {mk: (float(mv) if mv is not None and np.isfinite(mv) else None)
                for mk, mv in v.items()}
            for k, v in all_results.items()
        },
    }
    out_path = res_dir / "eval_axis_a.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [05a] Saved -> {out_path}")

    return out


def _print_metrics(method: str, metrics: dict) -> None:
    print(f"       {method}:")
    for k, v in metrics.items():
        val = f"{v:.4f}" if v is not None and np.isfinite(float(v if v is not None else float('nan'))) else "N/A"
        print(f"         {k:<20} {val}")


def _print_comparison(results: list[dict]) -> None:
    methods = list(results[0]["methods"].keys())
    metrics = ["ICC", "MCC_coverage", "LD", "MCC_disent"]
    for bench_res in results:
        bench = bench_res["benchmark"]
        print(f"\n{'='*60}")
        print(f"  {bench} / {bench_res['classifier']}")
        print(f"  causal_channel={bench_res['causal_channel']}, parents={bench_res['causal_parents']}")
        print(f"\n  {'Method':<12}", end="")
        for m in metrics:
            print(f"  {m:>14}", end="")
        print()
        print("  " + "-" * (12 + 16 * len(metrics)))
        for method in methods:
            print(f"  {method:<12}", end="")
            for m in metrics:
                v = bench_res["methods"].get(method, {}).get(m)
                val = f"{v:.4f}" if v is not None else "N/A"
                print(f"  {val:>14}", end="")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="LinearSCMT", choices=list(BENCH_MAP))
    parser.add_argument("--classifier", default="lstm", choices=list(CLF_MAP))
    parser.add_argument("--all_benchmarks", action="store_true",
                        help="Run on all benchmarks that have trained models")
    args = parser.parse_args()

    if args.all_benchmarks:
        benchmarks = [b for b in BENCH_MAP if (MODEL_DIR / b / args.classifier).exists()]
    else:
        benchmarks = [args.benchmark]

    all_results = []
    for bench in benchmarks:
        print(f"\n[05a] === {bench} / {args.classifier} ===")
        r = evaluate_axis_a(bench, args.classifier)
        all_results.append(r)

    _print_comparison(all_results)


if __name__ == "__main__":
    main()
