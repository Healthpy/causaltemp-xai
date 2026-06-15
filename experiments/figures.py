"""Publication figures for the CausalTemp-XAI v0.1 benchmark.

Reads ``experiments/results.json`` (summary + provenance) and
``experiments/per_instance.csv`` (one row per method/instance) produced by
``run_all.py`` and emits three PNGs to ``experiments/figures/``:

* **Fig 1** — validity vs CF-faith(rollout) scatter, one point per instance,
  coloured by method, with per-method means overlaid. Shows the dissociation:
  high validity does not buy causal faithfulness.
* **Fig 2** — CF-faith distribution per method, drawn for **both** semantics
  (rollout | pearl) side by side, to make the rollout-vs-pearl contrast visible.
* **Fig 3** — rank correlation: Spearman ρ between each traditional metric
  (validity, proximity, sparsity) and CF-faith(rollout, hard) across methods.

Usage
-----
    uv run python experiments/figures.py --results experiments/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
FIG_DIR = EXP_DIR / "figures"

#: Stable per-method colours (one per CF family).
COLORS = {
    "Wachter": "#1f77b4",
    "DiCE": "#2ca02c",
    "CARLA": "#d62728",
}


def _load_per_instance(path: Path):
    """Return ``{column: np.ndarray}`` from the per-instance CSV."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty — run run_all.py first")
    cols = rows[0].keys()
    out = {}
    for c in cols:
        vals = [r[c] for r in rows]
        if c == "method":
            out[c] = np.array(vals)
        else:
            out[c] = np.array([float(v) for v in vals])
    return out


def _methods_in_order(method_col):
    seen = []
    for m in method_col:
        if m not in seen:
            seen.append(m)
    return seen


def fig1_scatter(data, methods, out_path):
    """Validity (jittered) vs CF-faith(rollout, soft) per instance."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        mask = data["method"] == m
        x = data["validity"][mask] + rng.uniform(-0.06, 0.06, size=int(mask.sum()))
        y = data["cf_faith_rollout_soft"][mask]
        color = COLORS.get(m, None)
        ax.scatter(x, y, alpha=0.5, s=35, color=color, label=m, edgecolors="none")
        # Per-method mean marker.
        ax.scatter(
            data["validity"][mask].mean(),
            y.mean(),
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
    ax.legend(title="method")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig2_distributions(data, methods, out_path):
    """Violin of CF-faith(soft) per method, rollout | pearl panels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (semantics, key) in zip(
        axes,
        [("rollout", "cf_faith_rollout_soft"), ("pearl", "cf_faith_pearl_soft")],
    ):
        series = [data[key][data["method"] == m] for m in methods]
        positions = np.arange(1, len(methods) + 1)
        parts = ax.violinplot(series, positions=positions, showmeans=True, widths=0.8)
        for body, m in zip(parts["bodies"], methods):
            body.set_facecolor(COLORS.get(m, "#888888"))
            body.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods)
        ax.set_title(f"CF-faith ({semantics}, soft)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("CF-faith (soft)")
    fig.suptitle("CF-faith distribution by method — rollout vs pearl semantics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig3_rank_correlation(data, methods, out_path):
    """Spearman ρ between traditional metrics and CF-faith(rollout, hard), across methods.

    Computed on the per-method mean of each metric. With only 3-4 methods this is
    underpowered (H4 is reported as preliminary, not confirmed) — annotated as such.
    """
    trad = ["validity", "proximity_l1", "sparsity"]
    faith = np.array(
        [data["cf_faith_rollout_hard"][data["method"] == m].mean() for m in methods]
    )
    rhos = []
    for t in trad:
        vec = np.array([data[t][data["method"] == m].mean() for m in methods])
        if np.allclose(vec, vec[0]) or np.allclose(faith, faith[0]):
            rho = np.nan
        else:
            rho, _ = spearmanr(vec, faith)
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(trad))
    plotted = [0.0 if np.isnan(r) else r for r in rhos]
    bars = ax.bar(xs, plotted, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.8)
    for x, r in zip(xs, rhos):
        label = "n/a" if np.isnan(r) else f"{r:+.2f}"
        ax.text(x, 0.02, label, ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["validity", "proximity_l1", "sparsity"])
    ax.set_ylabel("Spearman ρ vs CF-faith (rollout, hard)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(
        f"Rank correlation: traditional metrics vs CF-faith\n"
        f"(across {len(methods)} methods — preliminary / underpowered, H4)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the 3 benchmark figures.")
    parser.add_argument("--results", default=str(EXP_DIR / "results.json"))
    parser.add_argument("--per-instance", default=str(EXP_DIR / "per_instance.csv"))
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    args = parser.parse_args(argv)

    with open(args.results) as fh:
        json.load(fh)  # validates presence; per-instance drives the figures
    data = _load_per_instance(Path(args.per_instance))
    methods = _methods_in_order(data["method"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_scatter(data, methods, out_dir / "fig1_validity_vs_cffaith.png")
    fig2_distributions(data, methods, out_dir / "fig2_cffaith_distribution.png")
    fig3_rank_correlation(data, methods, out_dir / "fig3_rank_correlation.png")
    print(f"Wrote 3 figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
