"""Phase 08: self-graphing + Axis-B graph-error decomposition (H3).

Fits a *self-graphing* method on a *nonlinear* (MLP-mechanism) benchmark's data,
reads off its inferred lag-1 causal graph, and reports
``results/<config>/<method>/graph_error.json`` with:

(a) Axis-B graph-recovery of the inferred graph vs. ground truth (SHD/LagAcc/AUC);
(b) the **graph-error decomposition** the general plan calls for: how much
    CF-faith the oracle structural counterfactual loses when it is derived from
    the method's *inferred* graph instead of the true graph (``graph_error``),
    versus propagation error (``propagation_error``, ≡0 for the oracle); and
(c) per-CF-method **propagation error** — each existing CF method's real CFs
    scored against the true SCM, under both CF-faith semantics.

Methods (``--method``):
- ``dynotears`` (default) — DYNOTEARS (Pamfil et al., 2020, vendored McKinsey
  CausalNex): a classical temporal causal-discovery baseline that recovers the
  benchmark's near-linear lag-1 structure from **observational** data (AUC
  ~0.9). This is the load-bearing graph-aware method that gives the Axis-B
  decomposition real dynamic range and H3 a genuine, non-circular positive.
- ``citris`` — genuine vendored CITRIS (representation learning; needs
  intervention-labeled data). An honest *secondary* method: it does not
  identify at smoke scale (see ``methods/causal/citris.py``).

Only nonlinear presets (``mechanism_type == "mlp"``, e.g. ``smoke_nl`` /
``full_nl``) are supported — the graph-error decomposition needs an
:class:`MLPMechanism` to build the inferred-graph rollout.

Usage
-----
    uv run python experiments/08_citris_graph.py --config smoke_nl
    uv run python experiments/08_citris_graph.py --config smoke_nl --method citris
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmarks.interventional import (  # noqa: E402
    generate_interventional_sequences,
)
from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.methods.causal import CITRIS, DYNOTEARS  # noqa: E402
from causaltemp_xai.metrics.axis_b import compute_axis_b  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402
from causaltemp_xai.scm.intervention import derive_intervention_t  # noqa: E402
from experiments._common import (  # noqa: E402
    build_masked_mechanism,
    build_oracle_interventions,
    config_dir,
    dump_json,
)

ORACLE_SHIFT = 1.5  # must match build_oracle_interventions' default


def _mean_soft_cf_faith(X_sel, cfs, graph, mechanism, scorer) -> float:
    """Mean soft rollout CF-faith of ``cfs`` scored against ``(graph, mechanism)``.

    ``intervention_t`` is derived per instance from ``(x, x_cf)`` (mechanism-
    independent), so the true-graph and inferred-graph scorings use the same
    intervention point and differ only in the mechanism the CF is checked
    against.
    """
    vals = []
    for x, x_cf in zip(np.asarray(X_sel, dtype=float), np.asarray(cfs, dtype=float)):
        t = derive_intervention_t(x, x_cf)
        vals.append(scorer.score(x, x_cf, t, graph, mechanism)["soft"])
    return float(np.mean(vals)) if vals else float("nan")


def _per_method_propagation(cf_dir, graph, true_mech, rollout, pearl):
    """Propagation error of every persisted CF method's *actual* CFs.

    Each ``X_cf_<Method>.npy`` is scored against the **true** SCM; the method's
    ``propagation_error = 1 − cf_faith_vs_gt`` measures how far its CF fails to
    respect the true mechanism (its own failure, independent of any inferred
    graph). Reported under **both** CF-faith semantics — ``noiseless_rollout``
    and ``pearl_delta`` (``*_pearl`` keys) — because recourse variants target
    different semantics (``CARLARecourse`` is rollout-faithful by construction,
    ``PearlCARLARecourse`` pearl-faithful), so a single-semantics column
    understates whichever targets the other.

    Note: a *per-method* graph-error term is intentionally NOT reported. A
    graph-error attribution requires varying the graph the **CF is derived
    from** (as the oracle decomposition does); these methods never use the
    inferred graph, so scoring their fixed CFs against a swapped mechanism is
    not a coherent graph-error and was found to be numerical noise (it could go
    negative). The oracle row is the sound graph-error reference.

    Returns a list of per-method dicts (empty if no CF arrays are present).
    """
    x_sel_path = cf_dir / "X_sel.npy"
    if not x_sel_path.exists():
        return []
    X_sel = np.load(x_sel_path)
    rows = []
    for cf_path in sorted(cf_dir.glob("X_cf_*.npy")):
        method = cf_path.stem[len("X_cf_"):]
        cfs = np.load(cf_path)
        r_gt = _mean_soft_cf_faith(X_sel, cfs, graph, true_mech, rollout)
        p_gt = _mean_soft_cf_faith(X_sel, cfs, graph, true_mech, pearl)
        rows.append({
            "method": method,
            "cf_faith_gt": r_gt,
            "propagation_error": float(1.0 - r_gt),
            "cf_faith_pearl_gt": p_gt,
            "propagation_error_pearl": float(1.0 - p_gt),
        })
    return rows


def _density_matched_adjacency(scores: np.ndarray, n_true: int) -> np.ndarray:
    """Binarise ``scores`` ``(k, k, L)`` keeping the top-``n_true`` off-diagonal
    entries (self-loops already zeroed in the scores). Ties broken by flat index."""
    adj = np.zeros_like(scores, dtype=int)
    if n_true <= 0:
        return adj
    flat = scores.reshape(-1)
    # Rank all entries; the diagonal is 0 in scores so it never wins a slot
    # unless n_true exceeds the number of positive-score edges.
    top = np.argsort(-flat)[:n_true]
    adj.reshape(-1)[top] = 1
    return adj


def _inferred_cf_faith(graph, mech, adj_pred, X_sel, oracle_ints, rollout) -> float:
    """Mean soft rollout CF-faith of the oracle CF *derived from* ``adj_pred``
    (a masked mechanism restricted to the inferred edges), scored against the
    true mechanism. Lower ⇒ more graph-induced divergence."""
    inf_mech = build_masked_mechanism(mech, adj_pred)
    vals = []
    for (t0, node, _true_cf), x in zip(oracle_ints, X_sel):
        value = float(x[t0, node]) + ORACLE_SHIFT
        inf_cf = structural_counterfactual(x, inf_mech, t0, node, value, noiseless=True)
        vals.append(rollout.score(x, inf_cf, t0, graph, mech)["soft"])
    return float(np.mean(vals))


def _corrupt_graph(true_bin: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Density-matched corruption of ``true_bin`` ``(k, k, L)``: replace a
    ``frac`` fraction of true edges with randomly chosen off-diagonal non-edges.
    ``frac=0`` returns the true graph; ``frac=1`` a random graph of the same
    density (recovery ≈ chance). Used to build the graph-quality ladder."""
    adj = true_bin.copy().astype(int)
    edges = list(zip(*np.nonzero(true_bin)))
    non_edges = [
        (i, j, l)
        for i in range(true_bin.shape[0])
        for j in range(true_bin.shape[1])
        for l in range(true_bin.shape[2])
        if i != j and true_bin[i, j, l] == 0
    ]
    n_swap = int(round(frac * len(edges)))
    if n_swap == 0 or not non_edges:
        return adj
    drop = rng.choice(len(edges), size=min(n_swap, len(edges)), replace=False)
    add = rng.choice(len(non_edges), size=min(n_swap, len(non_edges)), replace=False)
    for d in drop:
        adj[edges[d]] = 0
    for a in add:
        adj[non_edges[a]] = 1
    return adj


def _graph_quality_sweep(graph, mech, X_sel, oracle_ints, rollout, cf_faith_gt,
                         method_adj, method_auc, method_label, seed=0) -> list[dict]:
    """Graph-error across a controlled graph-quality ladder.

    Demonstrates the Axis-B decomposition *discriminates*: ``graph_error``
    (``cf_faith_gt − cf_faith_inferred``) must span ~0 for a good graph up to
    large for a random graph. Points: the true graph (0 by construction),
    progressively corrupted graphs (``frac`` of edges rewired), a fully random
    graph (high-error endpoint), and DYNOTEARS's *actual* recovered graph as the
    real-method anchor. Graph quality is reported as SHD-to-true and (for the
    corruption ladder) the corrupted fraction.
    """
    from causaltemp_xai.metrics.axis_b import shd, graph_auc

    true_bin = (graph > 0).astype(int)
    rng = np.random.default_rng(seed)
    rows = []

    def _point(label, adj, auc):
        cf_inf = _inferred_cf_faith(graph, mech, adj, X_sel, oracle_ints, rollout)
        rows.append({
            "label": label,
            "graph_auc": (None if auc is None else float(auc)),
            "shd": float(shd(true_bin, adj)),
            "cf_faith_inferred": cf_inf,
            "graph_error": float(cf_faith_gt - cf_inf),
        })

    # Controlled ladder: true -> increasingly corrupted -> random.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        adj = _corrupt_graph(true_bin, frac, rng)
        auc = graph_auc(true_bin, adj.astype(float)) if frac > 0 else 1.0
        _point(f"corrupt_frac={frac:g}", adj, auc)
    # Real-method anchor.
    _point(method_label, method_adj, method_auc)
    return rows


def run(
    config_name: str,
    out_dir,
    method: str = "dynotears",
    n_cf: int = 40,
    epochs: int = 80,
    intervention_prob: float = 0.3,
    sweep: bool = False,
    seed: int | None = None,
) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    if cfg.mechanism_type != "mlp":
        raise SystemExit(
            f"[08] self-graphing graph-error needs a nonlinear (mlp) config; "
            f"{cfg.name!r} is mechanism_type={cfg.mechanism_type!r}. "
            "Try --config smoke_nl."
        )

    data = load_dataset(cfg.name, out_dir=out_dir)
    graph, mech = data["graph"], data["mechanism"]
    k, _, L = graph.shape

    # 1. Fit the self-graphing method and read off its inferred lag-1 graph.
    #    DYNOTEARS (default) is a classical temporal causal-discovery baseline
    #    that recovers the benchmark's near-linear structure from OBSERVATIONAL
    #    data — the load-bearing graph-aware method for Axis B / H3. CITRIS is
    #    an honest secondary (representation-learning) method that needs
    #    intervention-labeled data and does not identify at smoke scale (see
    #    methods/causal/citris.py).
    X_all = data["X_train"]
    citris_meta = None
    if method == "dynotears":
        print(f"[08] fitting DYNOTEARS on {X_all.shape[0]} sequences (k={k}, p={L}) ...")
        model = DYNOTEARS(k=k, p=L).fit(X_all)
        _, scores = model.inferred_graph(max_lag=L)
    elif method == "citris":
        ds = generate_interventional_sequences(
            mech, k=k, L=L, T=X_all.shape[1], N=X_all.shape[0],
            seed=cfg.seed, intervention_prob=intervention_prob, mode="single",
        )
        print(
            f"[08] fitting CITRIS on {ds.X.shape[0]} interventional sequences "
            f"(k={k}, epochs={epochs}) ..."
        )
        model = CITRIS(k=k, max_epochs=epochs, seed=0).fit(ds.X, ds.targets)
        _, scores = model.inferred_graph(max_lag=L)
        citris_meta = {
            "source": "third_party/citris_repo (github.com/phlippe/CITRIS)",
            "epochs": epochs, "intervention_prob": intervention_prob,
            "intervention_mode": "single",
            "num_latents": int(model.model.num_latents),
            "n_train_sequences": int(ds.X.shape[0]),
        }
    else:
        raise SystemExit(f"[08] unknown --method {method!r}; use 'dynotears' or 'citris'")

    # Density-matched binarisation: keep the top-|E_true| off-diagonal edges by
    # score (standard fair graph-recovery practice — a fixed score threshold is
    # arbitrary and tends to over-densify). This gives an interpretable SHD and
    # a non-degenerate graph-error decomposition.
    n_true = int((graph > 0).sum())
    adj_pred = _density_matched_adjacency(scores, n_true)

    # 2. Graph-error decomposition: oracle CF via true graph vs inferred graph.
    #    The true-graph oracle CF is noiseless_rollout-faithful by construction
    #    (cf_faith_gt ~ 1); routing the same interventions through a mechanism
    #    restricted to CITRIS's inferred edges degrades faithfulness by exactly
    #    the graph error.
    X_sel = np.asarray(data["X_test"][:n_cf], dtype=float)
    inf_mech = build_masked_mechanism(mech, adj_pred)
    rollout = CFfaith(semantics="noiseless_rollout")

    # Use the *soft* (continuous) CF-faith: the hard score is a per-instance
    # 0/1 indicator that any edge difference zeroes, so it cannot express the
    # *degree* of graph-induced divergence the decomposition is meant to grade.
    oracle_ints = build_oracle_interventions(X_sel, mech)
    gt_soft, inf_soft = [], []
    for (t0, node, true_cf), x in zip(oracle_ints, X_sel):
        value = float(x[t0, node]) + ORACLE_SHIFT
        inf_cf = structural_counterfactual(x, inf_mech, t0, node, value, noiseless=True)
        gt_soft.append(rollout.score(x, true_cf, t0, graph, mech)["soft"])
        inf_soft.append(rollout.score(x, inf_cf, t0, graph, mech)["soft"])
    cf_faith_gt = float(np.mean(gt_soft))
    cf_faith_inferred = float(np.mean(inf_soft))

    # 3. Axis B (graph recovery + oracle decomposition) in one call. The oracle
    #    row is the perfect-propagator reference: propagation_error == 0, so its
    #    graph_error is the pure cost of the inferred graph.
    axis_b = compute_axis_b(
        adj_true_lagged=graph,
        adj_pred_lagged=adj_pred,
        score_matrix=scores,
        cf_faith_gt=cf_faith_gt,
        cf_faith_inferred=cf_faith_inferred,
    )

    # 4. Per-method propagation error for the existing CF methods (Phase 03
    #    outputs): their real CFs carry a genuine propagation_error (they do not
    #    respect the true SCM even given the true graph). No per-method
    #    graph-error is reported — see _per_method_propagation's docstring.
    res_dir = config_dir(cfg.name, "lstm")
    pearl = CFfaith(semantics="pearl_delta")
    method_rows = _per_method_propagation(res_dir / "cf", graph, mech, rollout, pearl)

    # 5. Graph-quality sweep (optional): show graph_error spans ~0 (good graph)
    #    to large (random graph), so the Axis-B decomposition is demonstrably
    #    discriminative rather than evaluated at a single near-perfect point.
    sweep_rows = None
    if sweep:
        sweep_rows = _graph_quality_sweep(
            graph, mech, X_sel, oracle_ints, rollout, cf_faith_gt,
            method_adj=adj_pred, method_auc=axis_b.get("AUC"),
            method_label=method, seed=cfg.seed,
        )

    out = {
        "provenance": {
            "config": cfg.as_dict(),
            "method": method,
            "n_cf": int(len(X_sel)),
            "dynotears": (
                {"source": "third_party/causalnex_repo (github.com/mckinsey/causalnex)",
                 "lambda_a": model.lambda_a, "lambda_w": model.lambda_w, "p": model.p}
                if method == "dynotears" else None
            ),
            "citris": citris_meta,
        },
        "axis_b": axis_b,
        "oracle_decomposition": {
            "cf_faith_gt": axis_b["cf_faith_gt"],
            "cf_faith_inferred": axis_b["cf_faith_inferred"],
            "graph_error": axis_b["graph_error"],
            "propagation_error": axis_b["propagation_error"],
        },
        "methods": method_rows,
        "graph_quality_sweep": sweep_rows,
        "inferred_edges": int(adj_pred.sum()),
        "true_edges": int((graph > 0).sum()),
    }

    out_dir_res = config_dir(cfg.name, method)
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_error.json", out)

    print(
        f"[08] {method} Axis B: SHD={axis_b['SHD']:.0f} LagAcc={axis_b['LagAcc']:.2f} "
        f"AUC={axis_b.get('AUC', float('nan')):.3f}"
    )
    print(
        f"[08] oracle (perfect propagator): graph_error={axis_b['graph_error']:+.3f} "
        f"propagation_error={axis_b['propagation_error']:+.3f}"
    )
    if method_rows:
        print(f"[08] per-method decomposition ({len(method_rows)} CF methods):")
        print(
            f"       {'method':<16} {'faith_gt':>9} {'prop_err':>9} "
            f"{'faith_gt(P)':>12} {'prop_err(P)':>12}"
        )
        for r in method_rows:
            print(
                f"       {r['method']:<16} {r['cf_faith_gt']:>9.3f} "
                f"{r['propagation_error']:>+9.3f} {r['cf_faith_pearl_gt']:>12.3f} "
                f"{r['propagation_error_pearl']:>+12.3f}"
            )
    else:
        print("[08] no persisted CF methods found (run experiments/03 first) — "
              "per-method decomposition skipped")
    if sweep_rows:
        print("[08] graph-quality sweep (graph_error vs graph quality):")
        print(f"       {'graph':<18} {'AUC':>6} {'SHD':>5} {'graph_error':>12}")
        for r in sweep_rows:
            auc = "  n/a" if r["graph_auc"] is None else f"{r['graph_auc']:.2f}"
            print(
                f"       {r['label']:<18} {auc:>6} {r['shd']:>5.0f} "
                f"{r['graph_error']:>+12.3f}"
            )
    print(f"[08] wrote {out_dir_res / 'graph_error.json'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-graphing (DYNOTEARS / CITRIS) + Axis-B graph-error "
                    "decomposition (H3)."
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--method", default="dynotears", choices=("dynotears", "citris"),
        help="self-graphing method (default: dynotears — the load-bearing "
             "graph-aware baseline; citris is the honest secondary).",
    )
    parser.add_argument("--n-cf", type=int, default=40, help="instances for the decomposition")
    parser.add_argument("--epochs", type=int, default=80, help="CITRIS training epochs (citris only)")
    parser.add_argument("--intervention-prob", type=float, default=0.3)
    parser.add_argument(
        "--sweep", action="store_true",
        help="also compute the graph-quality sweep (graph_error across a "
             "controlled true→random graph ladder + the method's real graph), "
             "demonstrating the decomposition's dynamic range.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Multi-seed replicate (M2): override the config seed via seeded_variant.",
    )
    args = parser.parse_args(argv)
    run(
        args.config, args.out_dir, method=args.method, n_cf=args.n_cf,
        epochs=args.epochs, intervention_prob=args.intervention_prob,
        sweep=args.sweep, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
