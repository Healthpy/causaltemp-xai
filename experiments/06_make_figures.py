"""Phase 06: Render figures from every ``results/*/*/per_instance.csv``.

Aggregates across every (benchmark, classifier) combination that Phases 04/05
have produced and emits three figures to ``results/figures/``:

* **Fig 1** -- validity vs CF-faith(rollout) scatter, one point per instance,
  coloured by method, with per-method means overlaid. Shows the dissociation:
  high validity does not buy causal faithfulness.
* **Fig 2** -- CF-faith distribution per method, rollout | pearl side by side.
* **Fig 3** -- rank correlation: Spearman rho between each traditional metric
  (validity, proximity, sparsity) and CF-faith(rollout, hard) across methods.

Usage
-----
    uv run python experiments/06_make_figures.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Mean of empty slice")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments._common import FIGURES_DIR, RESULTS_DIR  # noqa: E402

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render figures from results/*/*/per_instance.csv.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_all_per_instance(results_dir)
    methods = _methods_in_order(data["method"])
    print(f"[06] loaded {len(data['method'])} per-instance rows across {len(methods)} methods")

    fig1_scatter(data, methods, out_dir / "fig1_validity_vs_cffaith.png")
    fig2_distributions(data, methods, out_dir / "fig2_cffaith_distribution.png")
    fig3_rank_correlation(data, methods, out_dir / "fig3_rank_correlation.png")
    print(f"\n[06] wrote 3 figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
