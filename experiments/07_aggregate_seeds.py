"""Phase 07: Multi-seed replication + bootstrap-CI aggregation (M2, O2).

Runs Phases 01 -> 02 -> 03 -> 04 once per requested seed for a **linear**
config (the LSTM classifier + CF-method pipeline), each seed replicate built
via ``causaltemp_xai.config.seeded_variant`` -- which re-seeds *every*
seeded stage in one field: the SCM generator (graph + mechanism +
trajectories), the stratified train/val/test split, the LSTM's
initialisation/training, and (transitively, since it is a deterministic
function of the seeded ``X_test`` order) CF-instance selection. Each seed's
artifacts land under ``results/<config>_seed<seed>/lstm/`` so no seed
overwrites another's.

After all requested seeds have produced a ``per_instance.csv``, this phase
pools every method's per-instance metrics across seeds and reports a
hierarchical (seed-cluster) bootstrap 95% CI on every aggregated mean --
the plan's Standing Decision #4 ("every headline number carries a CI from M2
onward"). See ``causaltemp_xai/stats.py`` for why a *hierarchical* (not flat)
bootstrap is used, and ``experiments/_common.py::aggregate_across_seeds`` for
the pooling logic.

Usage
-----
    # Full pipeline per seed, then aggregate (smoke-scale validation):
    uv run python experiments/07_aggregate_seeds.py --config smoke --seeds 0 1 2 --n-cf 20

    # Re-aggregate only (seeds' phases 01-04 already ran):
    uv run python experiments/07_aggregate_seeds.py --config smoke --seeds 0 1 2 --skip-runs

Scope note: this phase targets the **classifier + CF-method** pipeline
(phases 01-04). The oracle positive control (phase 05) is classifier-free and
not wired into this orchestrator -- multi-seed oracle aggregation would reuse
the same ``aggregate_across_seeds`` machinery against
``results/<config>_seed<seed>/oracle/per_instance.csv`` if needed later.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import CONFIGS, get_config  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR  # noqa: E402
from experiments._common import (  # noqa: E402
    TABLES_DIR,
    aggregate_across_seeds,
    print_seed_aggregate_table,
    write_csv,
)

# Numbered phase modules are not valid dotted-import identifiers via a plain
# `import` statement (leading digit), but importlib handles them fine -- this
# reuses the exact same generate_one/train_one/run functions the CLI entry
# points call, so there is only one implementation of each phase.
_phase01 = importlib.import_module("experiments.01_generate_benchmarks")
_phase02 = importlib.import_module("experiments.02_train_classifiers")
_phase03 = importlib.import_module("experiments.03_run_cf_methods")
_phase04 = importlib.import_module("experiments.04_evaluate_axes")


def run_seed(config_name: str, seed: int, n_cf: int, out_dir, hparams: dict,
             methods_filter=None) -> None:
    """Run phases 01->04 for one seed replicate of ``config_name``."""
    print(f"\n=== [07] seed={seed}: phase 01 (generate) ===")
    _phase01.generate_one(config_name, out_dir, shift_noise=None, seed=seed)

    print(f"=== [07] seed={seed}: phase 02 (train LSTM) ===")
    _phase02.train_one(config_name, out_dir, seed=seed, **hparams)

    print(f"=== [07] seed={seed}: phase 03 (CF methods) ===")
    _phase03.run(config_name, n_cf, out_dir, methods_filter=methods_filter, seed=seed)

    print(f"=== [07] seed={seed}: phase 04 (evaluate axes) ===")
    _phase04.run(config_name, out_dir, seed=seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run phases 01-04 per seed, then pool with bootstrap 95% CIs."
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Seed values to replicate (M2 target: >=5 for a reported table; "
             "2-3 is a fast correctness-only smoke check).",
    )
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                         help="Subset of CF method names to run (all 6 if omitted).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--n-boot", type=int, default=2000,
        help="Bootstrap resamples for the aggregated CI table (default 2000 "
             "for fast iteration; use >=10000 for a final reported table).",
    )
    parser.add_argument(
        "--skip-runs", action="store_true",
        help="Skip phases 01-04 and only (re-)aggregate already-produced "
             "per-seed results (e.g. to re-run the CI table with a different "
             "--n-boot without regenerating data).",
    )
    # Classifier hyperparameters (mirrors 02_train_classifiers.py's CLI).
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args(argv)

    n_cf = args.n_cf or (20 if args.config.startswith("smoke") else 100)
    hparams = dict(
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
        lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience,
    )

    if not args.skip_runs:
        for seed in args.seeds:
            run_seed(args.config, seed, n_cf, args.out_dir, hparams,
                      methods_filter=args.methods)
    else:
        print("[07] --skip-runs: aggregating already-produced per-seed results only.")

    print(f"\n=== [07] aggregating {len(args.seeds)} seeds for '{args.config}' ===")
    rows = aggregate_across_seeds(
        args.config, args.seeds, classifier="lstm", n_boot=args.n_boot
    )
    out_path = TABLES_DIR / f"table_seed_aggregate_{args.config}_lstm.csv"
    write_csv(out_path, rows)
    print(f"[07] wrote {out_path}")
    print()
    print_seed_aggregate_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
