"""Phase 10: Tier-2 orchestration suite (M4i, `DECISIONS.md` 2026-08-06).

Runs the full/smoke Tier-2 real-data sequence across 3 UCR/UEA multivariate
datasets meeting `docs/general_plan.md` §6's selection criteria (k >= 3
channels, genuine classification task, moderate sequence length, no
missingness): `BasicMotions` (k=6, T=100), `RacketSports` (k=6, T=30),
`Epilepsy` (k=3, T=206) -- verified 2026-08-06 (M4i), measured not assumed.

Both causal-discovery methods now run on Tier 2, mirroring Tier 1's
"discovery methods run across all tiers" requirement -- but Tier 2 has **no
ground truth graph**, so the check is structural agreement only
(`run_cross_method_agreement_real`, `07b_discovered_graph_real.py`), never an
AUC-vs-truth claim the way Tier 1's is.

Per dataset:

1. `experiments.00_prepare_real_dataset.prepare` (download/cache if not
   already present, idempotent).
2. `experiments.06b_horizon_sweep_real.run` (train LSTM, run the 5 graph-free
   CF methods, persist `X_sel.npy`/`X_cf_<Method>.npy`).
3. `experiments.07b_discovered_graph_real.run` (DYNOTEARS bootstrap ensemble +
   discovered-*mechanism* CF-faith -- unchanged, DYNOTEARS-only, M4g).
4. `experiments.07b_discovered_graph_real.run_cross_method_agreement_real`
   (DYNOTEARS-vs-PCMCIplus structural agreement -- new, M4i).

**Tier 2 has no multi-seed machinery** (unlike Tier 1's >=3-seed convention)
-- a real, disclosed gap, not silently claimed as parity.

Usage
-----
    uv run python experiments/10_tier2_real_suite.py --scale smoke
    uv run python experiments/10_tier2_real_suite.py --scale full
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.real_data import DEFAULT_REAL_DIR  # noqa: E402
from experiments._common import RESULTS_DIR, dump_json, write_csv  # noqa: E402

_phase00 = importlib.import_module("experiments.00_prepare_real_dataset")
_phase06b = importlib.import_module("experiments.06b_horizon_sweep_real")
_phase07b = importlib.import_module("experiments.07b_discovered_graph_real")

# Case-sensitive archive filenames at timeseriesclassification.com/aeon-toolkit/
# -- confirmed live (HTTP 200, successfully parsed) 2026-08-06 (M4i).
_ARCHIVE_NAME = {
    "basicmotions": "BasicMotions",
    "racketsports": "RacketSports",
    "epilepsy": "Epilepsy",
}
ALL_DATASETS = tuple(_ARCHIVE_NAME)
#: "smoke" = the one dataset already cached, no network required; "full" = all
#: three, two of which need a live fetch -- the genuinely open risk this
#: two-tier split is meant to isolate before committing to the larger run.
_SCALE_DATASETS = {
    "smoke": ("basicmotions",),
    "full": ALL_DATASETS,
}


def run_dataset(
    name: str,
    out_dir,
    skip_fetch: bool,
    skip_cf: bool,
    n_cf,
    target_class: int,
    ensemble_b: int,
    pc_alpha: float,
    threshold: float,
    seed: int,
) -> dict:
    """Run the full Tier-2 sequence for one dataset. Returns a summary record."""
    if not skip_fetch:
        _phase00.prepare(_ARCHIVE_NAME[name], out_dir=out_dir, seed=seed)
    else:
        print(f"[10] --skip-fetch: assuming {name!r} already cached at {out_dir}")

    cf_sel_path = ROOT / "results" / f"real_{name}" / "lstm" / "cf" / "X_sel.npy"
    if skip_cf and cf_sel_path.exists():
        print(f"[10] --skip-cf: reusing existing persisted CF arrays for {name!r}")
    else:
        _phase06b.run(name=name, out_dir=out_dir, n_cf=n_cf, target_class=target_class, seed=seed)

    _phase07b.run(name=name, out_dir=out_dir, ensemble_b=ensemble_b, seed=seed, threshold=threshold)
    _phase07b.run_cross_method_agreement_real(
        name=name, out_dir=out_dir, pc_alpha=pc_alpha, threshold=threshold, seed=seed
    )

    summary_path = ROOT / "results" / f"real_{name}" / "summary.json"
    cf_faith_path = ROOT / "results" / f"real_{name}" / "dynotears" / "cf_faith_discovered.json"
    cma_path = ROOT / "results" / f"real_{name}" / "cross_method_agreement" / "graph_agreement.json"
    meta_path = Path(out_dir) / name / "meta.json"
    with open(summary_path) as fh:
        summary = json.load(fh)
    with open(cf_faith_path) as fh:
        cf_faith = json.load(fh)
    with open(cma_path) as fh:
        cma = json.load(fh)
    with open(meta_path) as fh:
        meta = json.load(fh)

    return {
        "dataset": name,
        "k": meta["shapes"]["X_train"][-1],
        "n_classes": summary.get("n_classes"),
        "test_acc": summary.get("test_acc"),
        "shd_between_methods": cma["shd_between_methods"],
        "cf_faith_discovered_rollout_by_method": {
            m["method"]: m["cf_faith_discovered_rollout_mean"] for m in cf_faith["methods"]
        },
        "note": cma["note"],
    }


def run(
    datasets: list[str],
    out_dir,
    skip_fetch: bool,
    skip_cf: bool,
    n_cf,
    target_class: int,
    ensemble_b: int,
    pc_alpha: float,
    threshold: float,
    seed: int,
) -> None:
    records = []
    for name in datasets:
        print(f"\n=== [10] dataset={name!r} ===")
        records.append(
            run_dataset(
                name,
                out_dir,
                skip_fetch,
                skip_cf,
                n_cf,
                target_class,
                ensemble_b,
                pc_alpha,
                threshold,
                seed,
            )
        )

    out_root = RESULTS_DIR / "tier2_suite"
    out_root.mkdir(parents=True, exist_ok=True)
    dump_json(out_root / "summary.json", {"datasets": records})

    table_rows = []
    for r in records:
        for method, cf_faith in r["cf_faith_discovered_rollout_by_method"].items():
            table_rows.append(
                {
                    "dataset": r["dataset"],
                    "k": r["k"],
                    "n_classes": r["n_classes"],
                    "test_acc": r["test_acc"],
                    "method": method,
                    "cf_faith_discovered_rollout_mean": cf_faith,
                    "shd_between_methods": r["shd_between_methods"],
                }
            )
    write_csv(out_root / "table_method_suitability.csv", table_rows)

    print(f"\n[10] wrote {out_root / 'summary.json'}")
    print(f"[10] wrote {out_root / 'table_method_suitability.csv'}")
    print()
    for r in records:
        print(
            f"{r['dataset']:<14} k={r['k']} n_classes={r['n_classes']} "
            f"test_acc={r['test_acc']:.3f} shd_between_methods={r['shd_between_methods']:.0f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scale", default="smoke", choices=("smoke", "full", "both"))
    parser.add_argument("--datasets", nargs="+", default=None, choices=ALL_DATASETS)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-cf", action="store_true")
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--ensemble-b", type=int, default=5)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(DEFAULT_REAL_DIR))
    args = parser.parse_args(argv)

    if args.datasets:
        datasets = list(dict.fromkeys(args.datasets))  # de-dupe, preserve order
    elif args.scale == "both":
        datasets = list(ALL_DATASETS)
    else:
        datasets = list(_SCALE_DATASETS[args.scale])

    run(
        datasets,
        args.out_dir,
        args.skip_fetch,
        args.skip_cf,
        args.n_cf,
        args.target_class,
        args.ensemble_b,
        args.pc_alpha,
        args.threshold,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
