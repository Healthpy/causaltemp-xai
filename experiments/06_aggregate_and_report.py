"""Phase 06: post-hoc aggregation and publication artifacts.

The one phase that reads what Phases 01-05 wrote and turns it into the two
artifact families a paper needs. It runs *no* explainer and defines *no* metric
of its own -- every number here is pooled or plotted from an existing
``results/<config>/<classifier>/per_instance.csv``.

Two distinct reports, one per subcommand:

* ``seeds``   -> ``results/tables/table_seed_aggregate_<config>_lstm.csv``
  **Multi-seed replication + bootstrap CIs (M2, O2).** Runs Phases 01 -> 02 ->
  03 -> 04 once per requested seed, each replicate built via
  :func:`causaltemp_xai.config.seeded_variant` -- which re-seeds *every* seeded
  stage in one field: the SCM generator (graph + mechanism + trajectories), the
  stratified train/val/test split, the LSTM's initialisation/training, and
  (transitively, since it is a deterministic function of the seeded ``X_test``
  order) CF-instance selection. Each seed's artifacts land under
  ``results/<config>_seed<seed>/lstm/`` so no seed overwrites another's. Every
  method's per-instance metrics are then pooled across seeds and reported with
  a hierarchical (seed-cluster) bootstrap 95% CI -- the plan's Standing
  Decision #4 ("every headline number carries a CI from M2 onward"). See
  ``causaltemp_xai/stats.py`` for why a *hierarchical* (not flat) bootstrap is
  used, and ``experiments/_common.py::aggregate_across_seeds`` for the pooling.

* ``figures`` -> ``results/figures/*.png``
  **Cross-run figures.** Aggregates every (benchmark, classifier) combination
  Phases 04/05 have produced and emits three figures:

  - **Fig 1** -- validity vs CF-faith(rollout) scatter, one point per instance,
    coloured by method, with per-method means overlaid. Shows the dissociation:
    high validity does not buy causal faithfulness.
  - **Fig 2** -- CF-faith distribution per method, rollout | pearl side by side.
  - **Fig 3** -- rank correlation: Spearman rho between each traditional metric
    (validity, proximity, sparsity) and CF-faith(rollout, hard) across methods.

``all`` runs ``seeds`` then ``figures``, so the tables and the figures are
generated from the same freshly-pooled results.

Usage
-----
    # Full pipeline per seed, then pool with bootstrap CIs:
    uv run python experiments/06_aggregate_and_report.py seeds --config smoke --seeds 0 1 2 --n-cf 20

    # Re-aggregate only (seeds' phases 01-04 already ran):
    uv run python experiments/06_aggregate_and_report.py seeds --config smoke --seeds 0 1 2 --skip-runs

    # Render figures from whatever results/ already contains:
    uv run python experiments/06_aggregate_and_report.py figures

    # Both, in order:
    uv run python experiments/06_aggregate_and_report.py all --config smoke --seeds 0 1 2

Scope note: the ``seeds`` subcommand targets the **classifier + CF-method**
pipeline (phases 01-04). The oracle positive control (Phase 05) is
classifier-free and not wired into this orchestrator -- multi-seed oracle
aggregation would reuse the same ``aggregate_across_seeds`` machinery against
``results/<config>_seed<seed>/oracle/per_instance.csv`` if needed later.

"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Mean of empty slice")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import CONFIGS  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR  # noqa: E402
from experiments._common import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
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

COLORS = {
    "CARLA": "#d62728",
    "PearlCARLA": "#ff9896",
    "CftsWachter": "#17becf",
    "CftsCOMTE": "#2ca02c",
    "CftsConfeti": "#9467bd",
    "CftsCounts": "#8c564b",
    "CftsCels": "#e377c2",
    "OracleCF-Pearl": "#7f7f7f",
    "OracleCF-Rollout": "#bcbd22",
}


# ---------------------------------------------------------------------------
# Report 1: multi-seed replication + bootstrap CIs
# ---------------------------------------------------------------------------


def run_seed(config_name: str, seed: int, n_cf: int, out_dir, hparams: dict,
             methods_filter=None) -> None:
    """Run phases 01->04 for one seed replicate of ``config_name``."""
    print(f"\n=== [06] seed={seed}: phase 01 (generate) ===")
    _phase01.generate_one(config_name, out_dir, shift_noise=None, seed=seed)

    print(f"=== [06] seed={seed}: phase 02 (train LSTM) ===")
    _phase02.train_one(config_name, out_dir, seed=seed, **hparams)

    print(f"=== [06] seed={seed}: phase 03 (CF methods) ===")
    _phase03.run(config_name, n_cf, out_dir, methods_filter=methods_filter, seed=seed)

    print(f"=== [06] seed={seed}: phase 04 (evaluate axes) ===")
    _phase04.run(config_name, out_dir, seed=seed)


def run_seeds_report(args) -> None:
    """Replicate across seeds (unless ``--skip-runs``) and write the CI table."""
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
        print("[06] --skip-runs: aggregating already-produced per-seed results only.")

    print(f"\n=== [06] aggregating {len(args.seeds)} seeds for '{args.config}' ===")
    rows = aggregate_across_seeds(
        args.config, args.seeds, classifier="lstm", n_boot=args.n_boot
    )
    out_path = TABLES_DIR / f"table_seed_aggregate_{args.config}_lstm.csv"
    write_csv(out_path, rows)
    print(f"[06] wrote {out_path}")
    print()
    print_seed_aggregate_table(rows)


# ---------------------------------------------------------------------------
# Report 2: cross-run figures
# ---------------------------------------------------------------------------


def _load_all_per_instance(results_dir: Path) -> dict:
    rows = []
    for fpath in sorted(results_dir.glob("*/*/per_instance.csv")):
        with open(fpath, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise SystemExit(
            f"no per_instance.csv found under {results_dir} -- run Phase 04/05 first"
        )
    cols = rows[0].keys()
    out = {}
    for c in cols:
        vals = [r[c] for r in rows]
        if c in ("method", "benchmark", "classifier"):
            out[c] = np.array(vals)
        else:
            out[c] = np.array([float(v) if v != "" else np.nan for v in vals])
    return out


def _methods_in_order(method_col):
    seen = []
    for m in method_col:
        if m not in seen:
            seen.append(m)
    return seen


def fig1_scatter(data, methods, out_path):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        mask = data["method"] == m
        valid = data["validity"][mask]
        if np.all(np.isnan(valid)):
            continue
        x = valid + rng.uniform(-0.06, 0.06, size=int(mask.sum()))
        y = data["cf_faith_rollout_soft"][mask]
        color = COLORS.get(m)
        ax.scatter(x, y, alpha=0.5, s=35, color=color, label=m, edgecolors="none")
        ax.scatter(
            np.nanmean(valid), np.nanmean(y), marker="D", s=140, color=color,
            edgecolors="black", linewidths=1.5, zorder=5,
        )
    ax.set_xlabel("Validity (per instance, jittered)")
    ax.set_ylabel("CF-faith (rollout, soft)")
    ax.set_title("Validity vs causal faithfulness\n(diamonds = per-method means)")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="method", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [fig] -> {out_path}")


def fig2_distributions(data, methods, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (semantics, key) in zip(
        axes, [("rollout", "cf_faith_rollout_soft"), ("pearl", "cf_faith_pearl_soft")],
    ):
        series = [data[key][data["method"] == m] for m in methods]
        positions = np.arange(1, len(methods) + 1)
        parts = ax.violinplot(series, positions=positions, showmeans=True, widths=0.8)
        for body, m in zip(parts["bodies"], methods):
            body.set_facecolor(COLORS.get(m, "#888888"))
            body.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"CF-faith ({semantics}, soft)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("CF-faith (soft)")
    fig.suptitle("CF-faith distribution by method -- rollout vs pearl semantics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [fig] -> {out_path}")


def fig3_rank_correlation(data, methods, out_path):
    trad = ["validity", "proximity_l1", "sparsity"]
    faith = np.array([np.nanmean(data["cf_faith_rollout_hard"][data["method"] == m]) for m in methods])
    rhos = []
    for t in trad:
        vec = np.array([np.nanmean(data[t][data["method"] == m]) for m in methods])
        if np.any(np.isnan(vec)) or np.allclose(vec, vec[0]) or np.allclose(faith, faith[0]):
            rho = np.nan
        else:
            rho, _ = spearmanr(vec, faith)
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(trad))
    plotted = [0.0 if np.isnan(r) else r for r in rhos]
    ax.bar(xs, plotted, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.8)
    for x, r in zip(xs, rhos):
        label = "n/a" if np.isnan(r) else f"{r:+.2f}"
        ax.text(x, 0.02, label, ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(trad)
    ax.set_ylabel("Spearman rho vs CF-faith (rollout, hard)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(
        f"Rank correlation: traditional metrics vs CF-faith\n"
        f"(across {len(methods)} methods -- preliminary / underpowered)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [fig] -> {out_path}")


def run_figures_report(args) -> None:
    """Render every cross-run figure from ``results/*/*/per_instance.csv``."""
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir_figures)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all_per_instance(results_dir)
    methods = _methods_in_order(data["method"])
    print(f"[06] loaded {len(data['method'])} per-instance rows across {len(methods)} methods")

    fig1_scatter(data, methods, out_dir / "fig1_validity_vs_cffaith.png")
    fig2_distributions(data, methods, out_dir / "fig2_cffaith_distribution.png")
    fig3_rank_correlation(data, methods, out_dir / "fig3_rank_correlation.png")
    print(f"\n[06] wrote 3 figures to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_seeds_args(p) -> None:
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Seed values to replicate (M2 target: >=5 for a reported table; "
             "2-3 is a fast correctness-only smoke check).",
    )
    p.add_argument("--n-cf", type=int, default=None)
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of CF method names to run (all 7 if omitted).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--n-boot", type=int, default=2000,
        help="Bootstrap resamples for the aggregated CI table (default 2000 "
             "for fast iteration; use >=10000 for a final reported table).",
    )
    p.add_argument(
        "--skip-runs", action="store_true",
        help="Skip phases 01-04 and only (re-)aggregate already-produced "
             "per-seed results (e.g. to re-run the CI table with a different "
             "--n-boot without regenerating data).",
    )
    # Classifier hyperparameters (mirrors 02_train_classifiers.py's CLI).
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)


def _add_figures_args(p) -> None:
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--out-dir-figures", default=str(FIGURES_DIR),
                   help="Figure output directory (default: results/figures).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 06: post-hoc aggregation (multi-seed bootstrap CI tables) "
                    "and publication figures.",
    )
    sub = parser.add_subparsers(dest="report", required=True)

    # NB: '%%' -- argparse runs help strings through %-formatting, so a literal
    # percent sign must be escaped or --help raises ValueError.
    p_seeds = sub.add_parser(
        "seeds", help="run phases 01-04 per seed, then pool with bootstrap 95%% CIs"
    )
    _add_seeds_args(p_seeds)

    p_figs = sub.add_parser(
        "figures", help="render figures from results/*/*/per_instance.csv"
    )
    _add_figures_args(p_figs)

    p_all = sub.add_parser("all", help="run 'seeds' then 'figures'")
    _add_seeds_args(p_all)
    _add_figures_args(p_all)

    args = parser.parse_args(argv)

    if args.report in ("seeds", "all"):
        run_seeds_report(args)
    if args.report in ("figures", "all"):
        if args.report == "all":
            print()
        run_figures_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
