"""
Figure generation for the benchmark paper (Weeks 11-12).

Produces three figures:
  Fig 1 — Scatter: Validity vs CF-faith, colour-coded by method
  Fig 2 — Boxplot: CF-faith distribution per method, separated by benchmark
  Fig 3 — Heatmap: Spearman ρ matrix (traditional metrics vs CF-faith)

Usage:
  python experiments/figures.py --results_dir results/ --out_dir results/figures/
"""

import argparse
import json
import pathlib
import numpy as np


# Matplotlib import with non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


METHOD_COLORS = {
    "wachter": "#e15759",
    "comte": "#4e79a7",
    "tsevo": "#f28e2b",
    "glacier": "#59a14f",
    "cels": "#bab0ac",
    "confetti": "#d37295",
    "carla": "#b07aa1",
    "citris": "#76b7b2",
    "icitris": "#edc948",
}
METHOD_ORDER = ["wachter", "comte", "tsevo", "glacier", "cels", "confetti", "carla", "citris", "icitris"]


def load_eval_results(results_dir):
    records = []
    for fpath in results_dir.rglob("eval_*.json"):
        with open(fpath) as f:
            records.append(json.load(f))
    return records


def load_rank_corr(results_dir):
    fpath = results_dir / "rank_correlation.json"
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)
    return None


def fig1_scatter_validity_vs_cffaith(records, out_dir):
    """
    Fig 1: Scatter plot — Validity (x) vs CF-faith (y), colour per method.
    Each point is one (benchmark × classifier × method) combination.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = set()
    for r in records:
        method = r["method"]
        validity = r["axis_c"].get("Validity")
        cf_faith = r["axis_c"].get("CF_faith_mean")
        if validity is None or cf_faith is None:
            continue
        color = METHOD_COLORS.get(method, "#aaaaaa")
        label = method if method not in plotted else None
        ax.scatter(validity, cf_faith, color=color, label=label,
                   s=90, edgecolors="white", linewidths=0.8, zorder=3)
        plotted.add(method)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="CF-faith = 0 (no gain over factual)")
    ax.set_xlabel("Validity (fraction flipping to target class)", fontsize=12)
    ax.set_ylabel("CF-faith (causal faithfulness, ↑ better)", fontsize=12)
    ax.set_title("Fig 1: Validity vs CF-faith by Method", fontsize=13, fontweight="bold")
    ax.legend(title="Method", fontsize=9, title_fontsize=10,
              loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = out_dir / "fig1_validity_vs_cffaith.pdf"
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] -> {out}")


def fig2_boxplot_cffaith_per_method(records, out_dir):
    """
    Fig 2: Boxplot of CF-faith distributions per method, grouped by benchmark.
    """
    benchmarks = sorted(set(r.get("benchmark", "unknown") for r in records))
    methods_present = sorted(set(r["method"] for r in records),
                             key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

    n_bench = len(benchmarks)
    fig, axes = plt.subplots(1, max(n_bench, 1), figsize=(5 * max(n_bench, 1), 5),
                             sharey=True)
    if n_bench == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        data_by_method = []
        labels = []
        colors = []
        for method in methods_present:
            vals = [r["axis_c"].get("CF_faith_mean") for r in records
                    if r.get("benchmark") == bench and r["method"] == method
                    and r["axis_c"].get("CF_faith_mean") is not None]
            if vals:
                data_by_method.append(vals)
                labels.append(method)
                colors.append(METHOD_COLORS.get(method, "#aaaaaa"))

        if not data_by_method:
            ax.set_title(f"{bench}\n(no data)", fontsize=10)
            continue

        bp = ax.boxplot(data_by_method, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(bench.replace("SCMT", " SCM-T"), fontsize=11, fontweight="bold")
        ax.set_ylabel("CF-faith (↑ better)" if ax == axes[0] else "", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 2: CF-faith Distribution per Method and Benchmark",
                 fontsize=13, fontweight="bold", y=1.02)
    out = out_dir / "fig2_cffaith_boxplot.pdf"
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] -> {out}")


def fig3_spearman_heatmap(rank_corr_data, out_dir):
    """
    Fig 3: Heatmap of Spearman ρ between traditional metrics and CF-faith.
    Rows = traditional metrics, Columns = methods (from per_method_spearman).
    """
    if rank_corr_data is None:
        print("  [fig] rank_correlation.json not found — skipping Fig 3")
        return

    per_method = rank_corr_data.get("per_method_spearman", {})
    overall = rank_corr_data.get("overall_spearman", {})
    if not per_method and not overall:
        print("  [fig] No rank correlation data — skipping Fig 3")
        return

    # Combine overall + per_method into a heatmap
    metrics = [k for k, v in overall.items()
               if isinstance(v, dict) and "rho" in v]
    if not metrics:
        print("  [fig] No overall Spearman data — skipping Fig 3")
        return

    methods = list(per_method.keys())
    col_labels = methods + ["overall"]

    rho_matrix = np.full((len(metrics), len(col_labels)), np.nan)
    for j, method in enumerate(methods):
        for i, metric in enumerate(metrics):
            entry = per_method.get(method, {}).get(metric)
            if isinstance(entry, dict) and "rho" in entry:
                rho_matrix[i, j] = entry["rho"]
    # Overall column
    for i, metric in enumerate(metrics):
        entry = overall.get(metric)
        if isinstance(entry, dict) and "rho" in entry:
            rho_matrix[i, -1] = entry["rho"]

    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 1.2 + 1.5), len(metrics) * 0.8 + 1.5))
    masked = np.ma.masked_invalid(rho_matrix)
    im = ax.imshow(masked, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")

    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(col_labels)):
            val = rho_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 0.6 else "white")

    ax.set_title("Fig 3: Spearman ρ — Traditional Metrics vs CF-faith",
                 fontsize=12, fontweight="bold", pad=12)

    out = out_dir / "fig3_spearman_heatmap.pdf"
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--out_dir", default="results/figures/")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_eval_results(results_dir)
    rank_corr = load_rank_corr(results_dir)

    print(f"[fig] Loaded {len(records)} evaluation records")

    if not records:
        print("[fig] No records found. Run 04_evaluate_axes.py first.")
        return

    fig1_scatter_validity_vs_cffaith(records, out_dir)
    fig2_boxplot_cffaith_per_method(records, out_dir)
    fig3_spearman_heatmap(rank_corr, out_dir)

    print(f"\n[fig] All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
