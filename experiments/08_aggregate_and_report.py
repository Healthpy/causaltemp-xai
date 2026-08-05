"""Phase 08: post-hoc aggregation and publication artifacts.

The one phase that reads what the earlier phases wrote and turns it into the
artifact families a paper needs. It runs *no* explainer and defines *no* metric
of its own -- every number here is pooled or plotted from an artifact some
earlier phase already produced.

Numbered last because it is the downstream **consumer**::

    01 -> 02 -> 03 -+-> 04 --------+
                    |              |
                    +-> 06 --------+--> 08   (horizon reads 06/horizon/summary.json)
                    |              |         (pns     reads 07/pns.json)
                    +-> 07 --------+         (figures reads 04+05 per_instance.csv)
    05  independent leaf -- classifier-free oracle control

...but it is *also* an upstream **driver**: ``seeds`` runs phases 01->04
in-process, once per seed. The number reflects its reporting role; it is not a
claim that it only ever runs last. (It was numbered 06 until 2026-08-04, which
put the aggregator *before* the two phases it consumes.)

Five reports, one per subcommand:

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
  - **Fig 4** -- do-complexity calibration (vector PDF).

  Sourced from a **depth-2** glob over ``results/*/*/per_instance.csv``, which
  deliberately excludes Phase 06's deeper ``.../horizon/per_instance.csv`` --
  see :func:`_load_all_per_instance`.

* ``pns``     -> ``results/tables/table_pns_do_complexity_<config>.csv``
  Pools Phase 07's ``pns.json`` / ``pns_schedule.json`` across seeds.

* ``horizon`` -> ``results/tables/table_horizon_<configA>_vs_<configB>.csv``
  Pools Phase 06's ``horizon/summary.json`` across seeds and configs. Also
  writes ``table_collapse_horizon_<configA>_vs_<configB>.csv`` (M4d) -- per
  method, the ``T - t0`` at which the seed-pooled validity curve crosses 0.5,
  bootstrapped over seeds (:func:`causaltemp_xai.stats.collapse_horizon_ci`).
  Named with the words swapped from the main horizon table specifically so it
  does **not** match ``fig5_horizon_decay``'s ``table_horizon_*.csv`` glob --
  it has different columns (no ``T_minus_t0``/``validity`` per row) and would
  break that figure if picked up by the same pattern.

* ``graph_quality`` -> ``results/tables/table_graph_quality_<configA>_vs_<configB>_vs_....csv``
  Pools Phase 07's ``dynotears/graph_error.json`` (``--sweep``) across seeds
  and configs (M4e). One row per (config, ladder point) with bootstrapped
  ``graph_error``/``frac_vacuous``, plus each config's per-CF-method
  ``propagation_error`` min/max (the "method-axis span" that makes a flat
  graph-error curve legible as a *contrast*, not a null result — see
  ``fig6_graph_quality_curve``).

``all`` runs ``seeds`` then ``figures``, so the tables and the figures are
generated from the same freshly-pooled results.

Usage
-----
    # Full pipeline per seed, then pool with bootstrap CIs:
    uv run python experiments/08_aggregate_and_report.py seeds --config smoke --seeds 0 1 2 --n-cf 20

    # Re-aggregate only (seeds' phases 01-04 already ran):
    uv run python experiments/08_aggregate_and_report.py seeds --config smoke --seeds 0 1 2 --skip-runs

    # Render figures from whatever results/ already contains:
    uv run python experiments/08_aggregate_and_report.py figures

    # Both, in order:
    uv run python experiments/08_aggregate_and_report.py all --config smoke --seeds 0 1 2

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
import json
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
    check_provenance,
    print_provenance_warning,
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


def run_seed(
    config_name: str, seed: int, n_cf: int, out_dir, hparams: dict, methods_filter=None
) -> None:
    """Run phases 01->04 for one seed replicate of ``config_name``."""
    print(f"\n=== [08] seed={seed}: phase 01 (generate) ===")
    _phase01.generate_one(config_name, out_dir, shift_noise=None, seed=seed)

    print(f"=== [08] seed={seed}: phase 02 (train LSTM) ===")
    _phase02.train_one(config_name, out_dir, seed=seed, **hparams)

    print(f"=== [08] seed={seed}: phase 03 (CF methods) ===")
    _phase03.run(config_name, n_cf, out_dir, methods_filter=methods_filter, seed=seed)

    print(f"=== [08] seed={seed}: phase 04 (evaluate axes) ===")
    _phase04.run(config_name, out_dir, seed=seed)


def run_seeds_report(args) -> None:
    """Replicate across seeds (unless ``--skip-runs``) and write the CI table."""
    n_cf = args.n_cf or (20 if args.config.startswith("smoke") else 100)
    hparams = dict(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    if not args.skip_runs:
        for seed in args.seeds:
            run_seed(args.config, seed, n_cf, args.out_dir, hparams, methods_filter=args.methods)
    else:
        print("[08] --skip-runs: aggregating already-produced per-seed results only.")

    print(f"\n=== [08] aggregating {len(args.seeds)} seeds for '{args.config}' ===")
    rows = aggregate_across_seeds(args.config, args.seeds, classifier="lstm", n_boot=args.n_boot)
    out_path = TABLES_DIR / f"table_seed_aggregate_{args.config}_lstm.csv"
    write_csv(out_path, rows)
    print(f"[08] wrote {out_path}")

    # Same per-seed directories aggregate_across_seeds just read, reconstructed
    # the same way (via seeded_variant) so the provenance check is guaranteed
    # to look at the exact inputs the table came from, not an approximation.
    # NB: results always live under RESULTS_DIR (results/<config>/<classifier>/,
    # see _common.config_dir) regardless of --out-dir, which is the *data*
    # output directory (data/scm_t/ by default) -- not the results root.
    from causaltemp_xai.config import get_config, seeded_variant

    base_cfg = get_config(args.config)
    seed_dirs = [RESULTS_DIR / seeded_variant(base_cfg, s).name / "lstm" for s in args.seeds]
    print_provenance_warning(check_provenance(seed_dirs))

    print()
    print_seed_aggregate_table(rows)


# ---------------------------------------------------------------------------
# Report 2: cross-run figures
# ---------------------------------------------------------------------------


def _load_all_per_instance(results_dir: Path) -> dict:
    """Every main-pipeline ``per_instance.csv``, one row per (method, instance).

    The depth-2 glob is deliberate and must stay depth-2. Phase 06 writes a
    *third* level, ``<config>/<clf>/horizon/per_instance.csv``, which re-scores
    the **same instances** once per swept ``t0`` -- so folding it in here would
    count each of those instances four times over and blend distinct ``t0``
    regimes into one per-method distribution. The horizon sweep gets its own
    reading via the ``horizon`` subcommand and ``table_horizon_*.csv``.
    """
    rows = []
    for fpath in sorted(results_dir.glob("*/*/per_instance.csv")):
        with open(fpath, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no per_instance.csv found under {results_dir} -- run Phase 04/05 first")
    # Union, not rows[0].keys(): older per_instance.csv files can predate a
    # column added by a later metric fix (e.g. sparsity_channels/timepoints,
    # 2026-07-18) -- missing entries become NaN rather than a KeyError.
    cols: set = set()
    for r in rows:
        cols.update(r.keys())
    out = {}
    for c in cols:
        vals = [r.get(c, "") for r in rows]
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
            np.nanmean(valid),
            np.nanmean(y),
            marker="D",
            s=140,
            color=color,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
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
        axes,
        [("rollout", "cf_faith_rollout_soft"), ("pearl", "cf_faith_pearl_soft")],
    ):
        # Drop degenerate (NaN) instances before the KDE: violinplot's kernel
        # estimate returns garbage on NaN input (and warns from linalg.det).
        # A method with no scorable instances at all gets no violin and is
        # annotated instead -- an empty slot is honest, a fabricated shape or
        # a silent gap is not.
        raw = {m: data[key][data["method"] == m] for m in methods}
        clean = {m: v[~np.isnan(v)] for m, v in raw.items()}
        drawable = [m for m in methods if clean[m].size > 0]
        positions = np.arange(1, len(methods) + 1)
        pos_of = {m: p for m, p in zip(methods, positions)}

        if drawable:
            parts = ax.violinplot(
                [clean[m] for m in drawable],
                positions=[pos_of[m] for m in drawable],
                showmeans=True,
                widths=0.8,
            )
            for body, m in zip(parts["bodies"], drawable):
                body.set_facecolor(COLORS.get(m, "#888888"))
                body.set_alpha(0.6)
        for m in methods:
            if clean[m].size == 0:
                ax.text(
                    pos_of[m],
                    0.5,
                    "all\ndegen.",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#b00020",
                    style="italic",
                )
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
    faith = np.array(
        [np.nanmean(data["cf_faith_rollout_hard"][data["method"] == m]) for m in methods]
    )
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

    per_instance_paths = sorted(results_dir.glob("*/*/per_instance.csv"))
    print_provenance_warning(check_provenance([p.parent for p in per_instance_paths]))

    data = _load_all_per_instance(results_dir)
    methods = _methods_in_order(data["method"])
    print(f"[08] loaded {len(data['method'])} per-instance rows across {len(methods)} methods")

    fig1_scatter(data, methods, out_dir / "fig1_validity_vs_cffaith.png")
    fig2_distributions(data, methods, out_dir / "fig2_cffaith_distribution.png")
    fig3_rank_correlation(data, methods, out_dir / "fig3_rank_correlation.png")
    fig4_do_complexity_calibration(out_dir / "fig4_do_complexity_calibration.pdf")
    n = 4
    if fig5_horizon_decay(TABLES_DIR, out_dir / "fig5_horizon_decay.png"):
        n += 1
    if fig6_graph_quality_curve(TABLES_DIR, out_dir / "fig6_graph_quality_curve.pdf"):
        n += 1
    print(f"\n[08] wrote {n} figures to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_seeds_args(p) -> None:
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed values to replicate (M2 target: >=5 for a reported table; "
        "2-3 is a fast correctness-only smoke check).",
    )
    p.add_argument("--n-cf", type=int, default=None)
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of CF method names to run (all 7 if omitted).",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap resamples for the aggregated CI table (default 2000 "
        "for fast iteration; use >=10000 for a final reported table).",
    )
    p.add_argument(
        "--skip-runs",
        action="store_true",
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


# ---------------------------------------------------------------------------
# Report 3: PNS pooled across seeds, single-slice vs schedule reading (M2b)
# ---------------------------------------------------------------------------

#: Columns pooled across seeds for the PNS table. `delta_trajectory` is
#: included but carries no information in schedule mode (it is 0 by
#: construction there) — it is reported precisely so that the reader can see
#: that, rather than being shown only the reading that flatters the metric.
_PNS_COLUMNS = (
    "A_model_proposed",
    "B_model_oracle",
    "C_world_oracle",
    "delta_total",
    "delta_trajectory",
    "delta_outcome",
    "do_complexity_mean",
)


def run_pns_report(args) -> None:
    """Pool ``pns.json`` / ``pns_schedule.json`` across seeds with bootstrap CIs.

    The two readings are printed **side by side** on purpose. Neither is "the"
    answer: the single-slice reading inflates ``delta_trajectory`` for any
    method that edits densely, and the schedule reading drives it to 0 by
    construction (RISK-18). What discriminates the methods is ``D`` — how many
    timesteps the proposal must declare as ``do()`` before the mechanism can
    produce it — read alongside ``delta_outcome``, which the schedule reading
    leaves intact.
    """
    from causaltemp_xai.stats import bootstrap_ci

    results_dir = Path(args.results_dir)
    # An empty --seeds means "the base config directory", matching
    # run_horizon_report. Without this the path is seed-mandatory, so an
    # unseeded single run (results/smoke/lstm/pns.json, which is what
    # `07 --method pns` writes when given no --seed) could never be read and
    # the command exited "no PNS results found" for a file sitting on disk.
    seeds = args.seeds if args.seeds else [None]
    rows = []

    for mode, fname in (("single_slice", "pns.json"), ("schedule", "pns_schedule.json")):
        per_method: dict[str, dict[str, list]] = {}
        for seed in seeds:
            name = args.config if seed is None else f"{args.config}_seed{seed}"
            path = results_dir / name / "lstm" / fname
            if not path.exists():
                print(f"[08] missing {path} — skipping")
                continue
            payload = json.loads(path.read_text())
            for method, out in payload.get("PS", {}).items():
                slot = per_method.setdefault(method, {c: [] for c in _PNS_COLUMNS})
                for col in _PNS_COLUMNS:
                    slot[col].append(out.get(col, float("nan")))
        for method, cols in sorted(per_method.items()):
            row = {"config": args.config, "mode": mode, "method": method, "n_seeds": len(seeds)}
            for col, vals in cols.items():
                clean = [v for v in vals if v == v]
                if not clean:
                    row[col] = float("nan")
                    row[f"{col}_lo"] = row[f"{col}_hi"] = float("nan")
                    continue
                res = bootstrap_ci(clean, n_boot=args.n_boot)
                row[col] = res.mean
                row[f"{col}_lo"], row[f"{col}_hi"] = res.ci_lo, res.ci_hi
            rows.append(row)

    if not rows:
        raise SystemExit(f"[08] no PNS results found for {args.config!r} under {results_dir}")

    out_path = TABLES_DIR / f"table_pns_do_complexity_{args.config}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[08] PNS pooled over {len(seeds)} seeds — {args.config}")
    print(f"{'mode':<13} {'method':<14} {'A':>6} {'C':>6} {'d_total':>8} {'d_traj':>7} {'D':>7}")
    for r in rows:
        fmt = lambda v: "  nan" if v != v else f"{v:+.2f}"  # noqa: E731
        print(
            f"{r['mode']:<13} {r['method']:<14} {fmt(r['A_model_proposed']):>6} "
            f"{fmt(r['C_world_oracle']):>6} {fmt(r['delta_total']):>8} "
            f"{fmt(r['delta_trajectory']):>7} {r['do_complexity_mean']:>7.1f}"
        )
    print(f"[08] wrote {out_path}")


def run_horizon_report(args) -> None:
    """Pool Phase-08 horizon sweeps across seeds; two configs side by side (H8c).

    The point of the side-by-side is that a horizon curve is only attributable
    to a *distance* if two configs that share `T` but differ in where the label
    sits are shown together. Read down the `T-t0` column: if the two configs
    disagree at the same `T-t0`, decay is not a function of `T-t0`.
    """
    from causaltemp_xai.stats import bootstrap_ci, collapse_horizon_ci

    results_dir = Path(args.results_dir)
    # An empty --seeds means "the base config directory", which is how the
    # single-run smoke sweeps are laid out (results/smoke/, not smoke_seed0/).
    seeds = args.seeds if args.seeds else [None]
    rows = []
    collapse_rows = []
    for config in args.configs:
        per_cell: dict[tuple, dict[str, list]] = {}
        t_label_seen = set()
        # Per-(method, seed) validity-by-horizon curve, for the collapse-horizon
        # estimate below. Keyed separately from per_cell because that dict
        # collects validity *across* seeds per cell, losing which seed each
        # value came from -- collapse_horizon_ci needs each seed's own
        # complete curve, not a pre-pooled column.
        curves: dict[tuple, dict[float, float]] = {}
        for seed in seeds:
            name = config if seed is None else f"{config}_seed{seed}"
            path = results_dir / name / "lstm" / "horizon" / "summary.json"
            if not path.exists():
                print(f"[08] missing {path} — skipping")
                continue
            payload = json.loads(path.read_text())
            T = payload["T"]
            for r in payload["rows"]:
                # `t_label` is absent from sweeps run before the H8c wiring; for
                # those the label was terminal by definition, so derive it
                # rather than dropping the row.
                t_label = r.get("t_label", T - 1)
                t_label_seen.add(t_label)
                key = (r["method"], r["horizon"], r["t0"], t_label)
                slot = per_cell.setdefault(
                    key,
                    {"validity": [], "ps_C_world_oracle": [], "ps_delta_total": []},
                )
                for col in slot:
                    v = r.get(col)
                    if v is not None:
                        slot[col].append(v)
                if r.get("validity") is not None:
                    curves.setdefault((r["method"], seed), {})[r["horizon"]] = r["validity"]
        for (method, h, t0, t_label), cols in sorted(per_cell.items(), key=lambda kv: kv[0][1]):
            row = {
                "config": config,
                "method": method,
                "T_minus_t0": h,
                "t0": t0,
                "t_label": t_label,
                "label_horizon": t_label - t0,
                "n_seeds": len(cols["validity"]),
            }
            for col, vals in cols.items():
                clean = [v for v in vals if v == v]
                if not clean:
                    row[col] = row[f"{col}_lo"] = row[f"{col}_hi"] = float("nan")
                    continue
                res = bootstrap_ci(clean, n_boot=args.n_boot)
                row[col], row[f"{col}_lo"], row[f"{col}_hi"] = res.mean, res.ci_lo, res.ci_hi
            rows.append(row)

        # Collapse horizon per method (M4d): the T-t0 at which the seed-pooled
        # validity curve crosses 0.5, bootstrapped over seeds. Only seeds whose
        # curve covers every horizon this method was swept at are used --
        # collapse_horizon_ci assumes a complete, aligned grid per seed
        # (module docstring), and a partial curve would silently misalign
        # which horizon each validity value belongs to.
        methods_in_curves = sorted({m for m, _s in curves})
        for method in methods_in_curves:
            per_seed = {s: c for (m, s), c in curves.items() if m == method}
            all_horizons = sorted({h for c in per_seed.values() for h in c})
            complete_seeds = [s for s, c in per_seed.items() if all(h in c for h in all_horizons)]
            if len(all_horizons) < 2 or not complete_seeds:
                continue
            validity_by_seed = [[per_seed[s][h] for h in all_horizons] for s in complete_seeds]
            cr = collapse_horizon_ci(
                all_horizons, validity_by_seed, threshold=0.5, n_boot=args.n_boot
            )
            collapse_rows.append(
                {
                    "config": config,
                    "method": method,
                    "threshold": 0.5,
                    "collapse_horizon": cr.horizon,
                    "collapse_horizon_lo": cr.ci_lo,
                    "collapse_horizon_hi": cr.ci_hi,
                    "n_seeds": cr.n_seeds,
                    "frac_boot_crossed": cr.frac_boot_crossed,
                    "horizons_swept": ",".join(str(int(h)) for h in all_horizons),
                }
            )

    if not rows:
        raise SystemExit(f"[08] no horizon results found for {args.configs} under {results_dir}")

    out_path = TABLES_DIR / f"table_horizon_{'_vs_'.join(args.configs)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[08] horizon pooled over {len(seeds)} seed(s)")
    print(
        f"{'config':<22}{'method':<13}{'T-t0':>6}{'t0':>5}{'t_l-t0':>8}"
        f"{'validity [95% CI]':>26}{'C_world':>10}"
    )
    for r in sorted(rows, key=lambda r: (r["method"], r["T_minus_t0"], r["config"])):
        v, lo, hi = r["validity"], r["validity_lo"], r["validity_hi"]
        ci = "nan" if v != v else f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
        c = r["ps_C_world_oracle"]
        print(
            f"{r['config']:<22}{r['method']:<13}{r['T_minus_t0']:>6}{r['t0']:>5}"
            f"{r['label_horizon']:>8}{ci:>26}{c:>10.2f}"
        )
    print(f"[08] wrote {out_path}")

    if collapse_rows:
        collapse_path = TABLES_DIR / f"table_collapse_horizon_{'_vs_'.join(args.configs)}.csv"
        with open(collapse_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(collapse_rows[0]))
            writer.writeheader()
            writer.writerows(collapse_rows)
        print("\n[08] collapse horizon (T-t0 where validity crosses 0.5, bootstrapped over seeds):")
        print(f"{'config':<22}{'method':<13}{'horizon [95% CI]':>26}{'frac crossed':>14}")
        for r in sorted(collapse_rows, key=lambda r: (r["config"], r["method"])):
            h, lo, hi = r["collapse_horizon"], r["collapse_horizon_lo"], r["collapse_horizon_hi"]
            txt = "never (in range)" if h != h else f"{h:.1f} [{lo:.1f}, {hi:.1f}]"
            print(f"{r['config']:<22}{r['method']:<13}{txt:>26}{r['frac_boot_crossed']:>14.2f}")
        print(f"[08] wrote {collapse_path}")


def _add_horizon_args(p) -> None:
    p.add_argument("--configs", nargs="+", required=True, help="base config names to compare")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="seed replicates to pool; pass with no values to read the unseeded base config",
    )
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--n-boot", type=int, default=10000)


# ---------------------------------------------------------------------------
# Report 5: graph-quality degradation curve pooled across seeds (M4e)
# ---------------------------------------------------------------------------


def run_graph_quality_report(args) -> None:
    """Pool Phase-07's ``dynotears --sweep`` graph-error ladder across seeds
    and configs (M4e — Bahri et al. §V direction 1).

    Reads each config's ``results/<config>_seed<N>/dynotears/graph_error.json``,
    pools ``graph_quality_sweep`` (one point per corruption fraction, plus the
    real ``dynotears`` anchor) with a bootstrap 95%% CI on ``graph_error`` and
    ``frac_vacuous`` per point, and separately pools each config's per-CF-method
    ``propagation_error`` (from the ``methods`` block) into a min/max span —
    the method-axis dynamic range that a flat graph-error curve must be shown
    against, or it reads as a null finding rather than a finding about
    *dissipation* (`ROADMAP.md` M4e).
    """
    from causaltemp_xai.stats import bootstrap_ci

    results_dir = Path(args.results_dir)
    seeds = args.seeds if args.seeds else [None]
    rows = []

    for config in args.configs:
        per_label: dict[str, dict[str, list]] = {}
        prop_errors: list[float] = []
        n_seeds_found = 0
        for seed in seeds:
            name = config if seed is None else f"{config}_seed{seed}"
            path = results_dir / name / "dynotears" / "graph_error.json"
            if not path.exists():
                print(f"[08] missing {path} — skipping")
                continue
            n_seeds_found += 1
            payload = json.loads(path.read_text())
            for point in payload["graph_quality_sweep"]:
                slot = per_label.setdefault(
                    point["label"], {"graph_error": [], "frac_vacuous": [], "shd": []}
                )
                slot["graph_error"].append(point["graph_error"])
                slot["frac_vacuous"].append(point["frac_vacuous"])
                slot["shd"].append(point["shd"])
            for m in payload.get("methods", []):
                pe = m.get("propagation_error")
                if pe is not None and pe == pe:  # exclude NaN (e.g. PearlCARLA/full_nl)
                    prop_errors.append(pe)

        if not per_label:
            print(f"[08] no graph-quality sweep found for {config!r} — skipping")
            continue

        prop_lo = min(prop_errors) if prop_errors else float("nan")
        prop_hi = max(prop_errors) if prop_errors else float("nan")

        for label, cols in per_label.items():
            # The controlled ladder's label doubles as its exact corrupt_frac;
            # the "dynotears" anchor has no fraction (it's the method's own
            # recovered graph) — kept as NaN, not 0 or 1, so it's never
            # mistaken for a controlled ladder point on the x-axis.
            corrupt_frac = (
                float(label.split("=")[1]) if label.startswith("corrupt_frac=") else float("nan")
            )
            row = {
                "config": config,
                "label": label,
                "corrupt_frac": corrupt_frac,
                "n_seeds": len(cols["graph_error"]),
                "shd_mean": float(np.mean(cols["shd"])),
                "propagation_error_min": prop_lo,
                "propagation_error_max": prop_hi,
            }
            for col in ("graph_error", "frac_vacuous"):
                vals = cols[col]
                res = bootstrap_ci(vals, n_boot=args.n_boot)
                row[col], row[f"{col}_lo"], row[f"{col}_hi"] = res.mean, res.ci_lo, res.ci_hi
            rows.append(row)

    if not rows:
        raise SystemExit(
            f"[08] no graph-quality results found for {args.configs} under {results_dir}"
        )

    out_path = TABLES_DIR / f"table_graph_quality_{'_vs_'.join(args.configs)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[08] graph-quality pooled over {len(seeds)} seed(s)")
    print(
        f"{'config':<20}{'label':<18}{'shd':>6}"
        f"{'graph_error [95% CI]':>26}{'method span [min,max]':>26}"
    )
    for r in sorted(rows, key=lambda r: (r["config"], r["shd_mean"])):
        e, lo, hi = r["graph_error"], r["graph_error_lo"], r["graph_error_hi"]
        span = f"[{r['propagation_error_min']:.3f}, {r['propagation_error_max']:.3f}]"
        print(
            f"{r['config']:<20}{r['label']:<18}{r['shd_mean']:>6.1f}"
            f"{f'{e:.4f} [{lo:.4f}, {hi:.4f}]':>26}{span:>26}"
        )
    print(f"[08] wrote {out_path}")


def _add_graph_quality_args(p) -> None:
    p.add_argument("--configs", nargs="+", required=True, help="base config names to compare")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="seed replicates to pool; pass with no values to read the unseeded base config",
    )
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--n-boot", type=int, default=10000)


def fig4_do_complexity_calibration(out_path, seed: int = 0) -> None:
    """Do-complexity vs a *known* amount of causal fidelity (M2b).

    The benchmark's graded-sensitivity evidence, and the one figure that is
    computed rather than measured: it sweeps an α-blend between an oracle
    counterfactual (`α = 0`, one genuine `do()`) and a direct rewrite of the
    same channel path (`α = 1`, mechanism bypassed) and plots `D` against `α`.

    Every other adversarial test in `docs/axis_metrics_report.md` is binary —
    construct a CF that should fail, assert it fails. None shows that a metric
    *tracks* causal fidelity rather than merely detecting its total absence.
    A reviewer is entitled to ask for that, and this is the answer.
    """
    from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
    from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual
    from causaltemp_xai.metrics.pns import do_complexity

    k, lag, T = 4, 1, 30
    rng = np.random.default_rng(seed)
    A_list = [rng.uniform(-0.3, 0.3, (k, k)) for _ in range(lag)]
    x = np.zeros((T, k))
    noise = rng.laplace(0, 0.05, (T, k))
    for t in range(lag, T):
        for step, A in enumerate(A_list, start=1):
            x[t] += A @ x[t - step]
        x[t] += noise[t]
    mech = LinearMechanism(A_list)

    oracle = structural_counterfactual(x, mech, t0=5, node=0, value=4.0, noiseless=False)
    direct = x.copy()
    direct[5:, 0] = oracle[5:, 0]  # same channel-0 path, no mechanism propagation

    alphas = np.linspace(0.0, 1.0, 11)
    ds = [do_complexity(x, (1 - a) * oracle + a * direct, mech) for a in alphas]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(alphas, ds, "o-", color="#2166AC", lw=1.6, ms=4)
    ax.axhline(1, color="#7B3294", ls="--", lw=1.0, label="oracle: single $do()$")
    ax.axhline(T - 5, color="#999999", ls=":", lw=1.0, label="every editable step")
    ax.set_xlabel(r"$\alpha$: oracle CF $\rightarrow$ direct rewrite")
    ax.set_ylabel("do-complexity $D$")
    ax.set_title("$D$ tracks causal fidelity", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"[08] wrote {out_path}  (D from {ds[0]} to {ds[-1]})")


def fig5_horizon_decay(tables_dir: Path, out_path) -> bool:
    """Validity as a function of the intervention-to-outcome distance ``T - t0``.

    Added 2026-08-04 to close the M2 DoD item "every validity and PNS figure
    reported with T - t0" -- none of fig1-4 carry a horizon axis. This is the
    one figure that should: it is the direct plot of H8/H8c's evidence, built
    from ``table_horizon_<configA>_vs_<configB>.csv`` (Phase 06 pooled across
    seeds with bootstrap CIs), not recomputed here. Only the CARLA family is
    swept (Phase 06's own scope note: sweeping Wachter-style unconstrained
    edits on the same axis as single-``do()`` methods would conflate two
    different failure modes -- see ``06_horizon_sweep.py``).

    Returns ``False`` (and writes nothing) if no horizon table exists yet, so
    a smoke-only ``figures`` run does not fail on missing paper-scale data.
    """
    paths = sorted(tables_dir.glob("table_horizon_*.csv"))
    if not paths:
        print("[08] fig5: no table_horizon_*.csv found -- skipped")
        return False

    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return False

    configs = sorted({r["config"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    linestyles = {c: ls for c, ls in zip(configs, ["-", "--", ":", "-."])}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in methods:
        color = COLORS.get(method, "#333333")
        for config in configs:
            pts = sorted(
                (
                    (
                        int(r["T_minus_t0"]),
                        float(r["validity"]),
                        float(r["validity_lo"]),
                        float(r["validity_hi"]),
                    )
                    for r in rows
                    if r["method"] == method and r["config"] == config
                ),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            los = [p[2] for p in pts]
            his = [p[3] for p in pts]
            ax.plot(
                xs,
                ys,
                linestyles[config],
                color=color,
                marker="o",
                ms=4,
                label=f"{method} ({config})",
            )
            ax.fill_between(xs, los, his, color=color, alpha=0.12, linewidth=0)

    ax.set_xlabel(r"Intervention-to-outcome distance $T - t_0$")
    ax.set_ylabel("Validity (pooled across seeds, 95% CI band)")
    ax.set_title("Recourse validity vs. horizon (H8/H8c evidence)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [fig] -> {out_path}")
    return True


#: Per-config colour/marker, distinct from COLORS (which is method-keyed) --
#: this figure's lines are one per *config*, not per method.
_GRAPH_QUALITY_STYLE = {
    "full_nl": ("#999999", "o"),  # dissipative baseline -- grey, deliberately unremarkable
    "smoke_spring": ("#2166AC", "s"),  # non-dissipative (M4c)
    "smoke_kuramoto": ("#B2182B", "^"),  # non-dissipative (M4c)
}


def fig6_graph_quality_curve(tables_dir: Path, out_path) -> bool:
    """Graph-error vs. graph quality, dissipative vs. non-dissipative (M4e).

    Two panels sharing a y-axis, because a flat `graph_error` curve alone
    reads as a null finding -- it is only a finding about *dissipation* once
    shown against the method-axis span the same decomposition produces
    (`ROADMAP.md` M4e: "~54x more dynamic range across methods than across
    graph quality" on `full_nl`). Left panel: `graph_error` against the
    controlled corruption fraction (0 = true graph, 1 = chance-level random
    graph of the same density) -- this, not raw SHD, is the x-axis, because
    it is exact and comparable across configs of different graph size,
    whereas absolute SHD scales with `k`. Right panel: each config's
    per-CF-method `propagation_error` range as a vertical bar on the *same*
    y-axis -- so whether the graph-quality curve's climb is small or large
    relative to ordinary method-to-method variation is visible in one glance,
    not left to a table lookup.

    Returns ``False`` (writes nothing) if no `table_graph_quality_*.csv`
    exists yet.
    """
    paths = sorted(tables_dir.glob("table_graph_quality_*.csv"))
    if not paths:
        print("[08] fig6: no table_graph_quality_*.csv found -- skipped")
        return False

    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return False

    configs = sorted(
        {r["config"] for r in rows},
        key=lambda c: list(_GRAPH_QUALITY_STYLE).index(c) if c in _GRAPH_QUALITY_STYLE else 99,
    )
    y_max = max(
        max(float(r["graph_error_hi"]) for r in rows),
        max(float(r["propagation_error_max"]) for r in rows),
    )

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(7.25, 3.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    for config in configs:
        color, marker = _GRAPH_QUALITY_STYLE.get(config, ("#333333", "x"))
        pts = sorted(
            (
                (
                    float(r["corrupt_frac"]),
                    float(r["graph_error"]),
                    float(r["graph_error_lo"]),
                    float(r["graph_error_hi"]),
                )
                for r in rows
                if r["config"] == config
                and r["corrupt_frac"] == r["corrupt_frac"]  # exclude NaN (the dynotears anchor)
            ),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        xs, ys, los, his = zip(*pts)
        ax_l.plot(xs, ys, "-", color=color, marker=marker, ms=5, lw=1.6, label=config)
        ax_l.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)

    ax_l.set_xlabel("Corruption fraction (0 = true graph, 1 = chance)")
    ax_l.set_ylabel("graph_error (CF-faith lost to the wrong graph)")
    ax_l.set_ylim(-0.02, y_max * 1.08)
    ax_l.legend(fontsize=7, frameon=False, loc="upper left")
    ax_l.grid(True, alpha=0.3)
    ax_l.set_title("Graph quality", fontsize=10)

    bar_x = np.arange(len(configs))
    for i, config in enumerate(configs):
        color, _ = _GRAPH_QUALITY_STYLE.get(config, ("#333333", "x"))
        lo = float(next(r["propagation_error_min"] for r in rows if r["config"] == config))
        hi = float(next(r["propagation_error_max"] for r in rows if r["config"] == config))
        ax_r.bar(i, hi - lo, bottom=lo, color=color, alpha=0.8, width=0.6)
    ax_r.set_xticks(bar_x)
    ax_r.set_xticklabels(configs, fontsize=7, rotation=20, ha="right")
    ax_r.set_ylim(-0.02, y_max * 1.08)
    ax_r.set_yticklabels([])
    ax_r.set_title("Method-axis span\n(propagation_error)", fontsize=9)
    ax_r.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Graph quality bites only where effects persist (M4e)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] -> {out_path}")
    return True


def _add_pns_args(p) -> None:
    p.add_argument("--config", required=True, help="base config name, e.g. 'full' or 'full_nl'")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="seed replicates to pool; pass with no values to read the unseeded base config",
    )
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--n-boot", type=int, default=10000)


def _add_figures_args(p) -> None:
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--out-dir-figures",
        default=str(FIGURES_DIR),
        help="Figure output directory (default: results/figures).",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 08: post-hoc aggregation (multi-seed bootstrap CI tables) "
        "and publication figures.",
    )
    sub = parser.add_subparsers(dest="report", required=True)

    # NB: '%%' -- argparse runs help strings through %-formatting, so a literal
    # percent sign must be escaped or --help raises ValueError.
    p_seeds = sub.add_parser(
        "seeds", help="run phases 01-04 per seed, then pool with bootstrap 95%% CIs"
    )
    _add_seeds_args(p_seeds)

    p_figs = sub.add_parser("figures", help="render figures from results/*/*/per_instance.csv")
    _add_figures_args(p_figs)

    p_pns = sub.add_parser(
        "pns", help="pool PNS across seeds; single-slice vs schedule reading (M2b)"
    )
    _add_pns_args(p_pns)

    p_hz = sub.add_parser(
        "horizon", help="pool Phase-08 horizon sweeps across seeds; compare configs (H8c)"
    )
    _add_horizon_args(p_hz)

    p_gq = sub.add_parser(
        "graph_quality",
        help="pool Phase-07 dynotears --sweep graph-error ladders across seeds/configs (M4e)",
    )
    _add_graph_quality_args(p_gq)

    p_all = sub.add_parser("all", help="run 'seeds' then 'figures'")
    _add_seeds_args(p_all)
    _add_figures_args(p_all)

    args = parser.parse_args(argv)

    if args.report == "pns":
        run_pns_report(args)
        return 0
    if args.report == "horizon":
        run_horizon_report(args)
        return 0
    if args.report == "graph_quality":
        run_graph_quality_report(args)
        return 0
    if args.report in ("seeds", "all"):
        run_seeds_report(args)
    if args.report in ("figures", "all"):
        if args.report == "all":
            print()
        run_figures_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
