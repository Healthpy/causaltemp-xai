"""Phase 07: auxiliary causal-method families outside the main 01-05 pipeline.

Phases 01-05 evaluate *counterfactual-generating* methods against a trained
LSTM. This phase covers the two method families that do not fit that shape --
they learn causal structure or a causal representation rather than producing a
counterfactual -- and scores each on the axis its output actually addresses.
Neither family is wired into the multi-seed orchestrator (Phase 06 ``seeds``),
and both are config-restricted; that is why they live here rather than in the
main pipeline.

Selected with ``--method``; each writes its own distinct report:

* ``dynotears`` / ``citris`` -> ``results/<config>/<method>/graph_error.json``
  **Self-graphing + Axis-B graph-error decomposition (H3).** Fits a
  self-graphing method on a *nonlinear* (MLP-mechanism) benchmark, reads off
  its inferred lag-1 causal graph, and reports:

  (a) Axis-B graph-recovery of the inferred graph vs. ground truth
      (SHD/LagAcc/LagF1/AUC);
  (b) the **graph-error decomposition** the general plan calls for: how much
      CF-faith the oracle structural counterfactual loses when it is derived
      from the method's *inferred* graph instead of the true graph
      (``graph_error``), versus propagation error (``propagation_error``, ==0
      for the oracle); and
  (c) per-CF-method **propagation error** -- each existing CF method's real CFs
      scored against the true SCM, under both CF-faith semantics.

  ``dynotears`` (default) is DYNOTEARS (Pamfil et al., 2020, vendored McKinsey
  CausalNex): a classical temporal causal-discovery baseline that recovers the
  benchmark's near-linear lag-1 structure from **observational** data (AUC
  ~0.9). This is the load-bearing graph-aware method that gives the Axis-B
  decomposition real dynamic range and H3 a genuine, non-circular positive.
  ``citris`` is genuine vendored CITRIS (representation learning; needs
  intervention-labeled data) -- an honest *secondary* method: it does not
  identify at smoke scale (see ``methods/causal/citris.py``).

  Only nonlinear presets (``mechanism_type == "mlp"``, e.g. ``smoke_nl`` /
  ``full_nl``) are supported -- the decomposition needs an
  :class:`MLPMechanism` to build the inferred-graph rollout.

* ``ivae`` -> ``results/<config>/ivae/icc.json``
  **Decoder-based Axis-A ICC** (latent-traversal Interventional Concept
  Consistency), the definitional ICC for representation methods. Trains an
  **iVAE** (a method that exposes an encoder + decoder) on a config's
  observational data, then scores each latent dimension with
  :func:`causaltemp_xai.metrics.axis_a.icc_latent` (matched-baseline, +/-delta,
  std-scaled) against the frozen LSTM. Because iVAE identifies factors only up
  to permutation, latent dims are first **aligned to ground-truth channels**
  via a Hungarian match on the |correlation| matrix; the causally-relevant
  concepts are the label channel (channel 0, since ``Y = 1[X_T^0 > theta]``)
  and its graph ancestors. ICC is then contrasted between
  causal-relevant-aligned and non-relevant-aligned latent dims.

  Per the PI metric review, this run reports the **sanity gates first** -- if
  the iVAE reconstruction does not preserve the classifier's decision
  (``recon_label_agreement`` low), ICC is *not interpretable* and that is
  reported honestly rather than forced. Magnitudes ``c in {1,2,3}`` are swept
  (pre-registered; no delta-hacking), and a permuted-assignment null is
  reported.

Usage
-----
    uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method dynotears
    uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method citris
    uv run python experiments/07_auxiliary_methods.py --config smoke --method ivae --epochs 60

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmarks.interventional import (  # noqa: E402
    generate_interventional_sequences,
)
from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual  # noqa: E402
from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.methods.causal import CITRIS, DYNOTEARS  # noqa: E402
from causaltemp_xai.methods.concept import iVAE  # noqa: E402
from causaltemp_xai.metrics.axis_a import icc_latent  # noqa: E402
from causaltemp_xai.metrics.axis_b import compute_axis_b  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402
from causaltemp_xai.scm.intervention import derive_intervention_t  # noqa: E402
from experiments._common import (  # noqa: E402
    build_masked_mechanism,
    build_oracle_interventions,
    config_dir,
    dump_json,
)

GRAPH_METHODS = ("dynotears", "citris")
ORACLE_SHIFT = 1.5  # must match build_oracle_interventions' default
RECON_AGREEMENT_GATE = 0.7  # below this, ICC is not interpretable

#: Per-method training-epoch defaults. The two families were separate phases
#: with separate ``--epochs`` defaults (CITRIS 80, iVAE 50); merging them under
#: one flag must not silently retune either, so ``--epochs`` defaults to None
#: and resolves here.
DEFAULT_EPOCHS = {"citris": 80, "ivae": 50}


# ---------------------------------------------------------------------------
# Report 1: self-graphing + Axis-B graph-error decomposition (dynotears/citris)
# ---------------------------------------------------------------------------


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
    ``propagation_error = 1 - cf_faith_vs_gt`` measures how far its CF fails to
    respect the true mechanism (its own failure, independent of any inferred
    graph). Reported under **both** CF-faith semantics -- ``noiseless_rollout``
    and ``pearl_delta`` (``*_pearl`` keys) -- because recourse variants target
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
    true mechanism. Lower => more graph-induced divergence."""
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
    density (recovery ~ chance). Used to build the graph-quality ladder."""
    adj = true_bin.copy().astype(int)
    edges = list(zip(*np.nonzero(true_bin)))
    non_edges = [
        (i, j, l)
        for i in range(true_bin.shape[0])
        for j in range(true_bin.shape[1])
        for l in range(true_bin.shape[2])
        if i != j and true_bin[i, j, l] == 0
    ]
    n_swap = round(frac * len(edges))
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
    (``cf_faith_gt - cf_faith_inferred``) must span ~0 for a good graph up to
    large for a random graph. Points: the true graph (0 by construction),
    progressively corrupted graphs (``frac`` of edges rewired), a fully random
    graph (high-error endpoint), and the method's *actual* recovered graph as
    the real-method anchor. Graph quality is reported as SHD-to-true and (for
    the corruption ladder) the corrupted fraction.
    """
    from causaltemp_xai.metrics.axis_b import graph_auc, shd

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


def run_graph_method(
    cfg,
    out_dir,
    method: str,
    n_cf: int = 40,
    epochs: int = 80,
    intervention_prob: float = 0.3,
    sweep: bool = False,
) -> None:
    """Self-graphing + Axis-B graph-error decomposition for ``dynotears``/``citris``."""
    if cfg.mechanism_type != "mlp":
        raise SystemExit(
            f"[07] self-graphing graph-error needs a nonlinear (mlp) config; "
            f"{cfg.name!r} is mechanism_type={cfg.mechanism_type!r}. "
            "Try --config smoke_nl."
        )

    data = load_dataset(cfg.name, out_dir=out_dir)
    graph, mech = data["graph"], data["mechanism"]
    k, _, L = graph.shape

    # 1. Fit the self-graphing method and read off its inferred lag-1 graph.
    #    DYNOTEARS (default) is a classical temporal causal-discovery baseline
    #    that recovers the benchmark's near-linear structure from OBSERVATIONAL
    #    data -- the load-bearing graph-aware method for Axis B / H3. CITRIS is
    #    an honest secondary (representation-learning) method that needs
    #    intervention-labeled data and does not identify at smoke scale (see
    #    methods/causal/citris.py).
    X_all = data["X_train"]
    citris_meta = None
    if method == "dynotears":
        print(f"[07] fitting DYNOTEARS on {X_all.shape[0]} sequences (k={k}, p={L}) ...")
        model = DYNOTEARS(k=k, p=L).fit(X_all)
        _, scores = model.inferred_graph(max_lag=L)
    else:  # citris -- the only other member of GRAPH_METHODS
        ds = generate_interventional_sequences(
            mech, k=k, L=L, T=X_all.shape[1], N=X_all.shape[0],
            seed=cfg.seed, intervention_prob=intervention_prob, mode="single",
        )
        print(
            f"[07] fitting CITRIS on {ds.X.shape[0]} interventional sequences "
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

    # Density-matched binarisation: keep the top-|E_true| off-diagonal edges by
    # score (standard fair graph-recovery practice -- a fixed score threshold is
    # arbitrary and tends to over-densify). This gives an interpretable SHD and
    # a non-degenerate graph-error decomposition.
    n_true = int((graph > 0).sum())
    adj_pred = _density_matched_adjacency(scores, n_true)

    # 2. Graph-error decomposition: oracle CF via true graph vs inferred graph.
    #    The true-graph oracle CF is noiseless_rollout-faithful by construction
    #    (cf_faith_gt ~ 1); routing the same interventions through a mechanism
    #    restricted to the inferred edges degrades faithfulness by exactly the
    #    graph error.
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
    #    graph-error is reported -- see _per_method_propagation's docstring.
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
            "n_cf": len(X_sel),
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
        f"[07] {method} Axis B: SHD={axis_b['SHD']:.0f} LagAcc={axis_b['LagAcc']:.2f} "
        f"AUC={axis_b.get('AUC', float('nan')):.3f}"
    )
    print(
        f"[07] oracle (perfect propagator): graph_error={axis_b['graph_error']:+.3f} "
        f"propagation_error={axis_b['propagation_error']:+.3f}"
    )
    if method_rows:
        print(f"[07] per-method decomposition ({len(method_rows)} CF methods):")
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
        print("[07] no persisted CF methods found (run experiments/03 first) -- "
              "per-method decomposition skipped")
    if sweep_rows:
        print("[07] graph-quality sweep (graph_error vs graph quality):")
        print(f"       {'graph':<18} {'AUC':>6} {'SHD':>5} {'graph_error':>12}")
        for r in sweep_rows:
            auc = "  n/a" if r["graph_auc"] is None else f"{r['graph_auc']:.2f}"
            print(
                f"       {r['label']:<18} {auc:>6} {r['shd']:>5.0f} "
                f"{r['graph_error']:>+12.3f}"
            )
    print(f"[07] wrote {out_dir_res / 'graph_error.json'}")


# ---------------------------------------------------------------------------
# Report 2: decoder-based Axis-A ICC (ivae)
# ---------------------------------------------------------------------------


def _ancestors_of(node: int, graph: np.ndarray) -> set[int]:
    """Transitive causal ancestors of ``node`` (incl. itself) in the lagged
    graph ``(k, k, L)`` where ``graph[i, j, l]==1`` means j causes i."""
    def parents(i):
        return {j for j in range(graph.shape[1]) if np.any(graph[i, j, :] != 0)}

    seen, stack = {node}, [node]
    while stack:
        cur = stack.pop()
        for p in parents(cur):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _align_latents_to_channels(Z: np.ndarray, F: np.ndarray):
    """Hungarian match on |Pearson corr| between latent dims ``Z`` (N, d_z) and
    ground-truth factors ``F`` (N, k). Returns (assign, mcc): ``assign[i]`` =
    channel matched to latent dim ``i``; ``mcc`` = mean matched |corr|."""
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Fc = (F - F.mean(0)) / (F.std(0) + 1e-12)
    corr = np.abs(Zc.T @ Fc) / Z.shape[0]  # (d_z, k)
    rows, cols = linear_sum_assignment(-corr)
    assign = {int(r): int(c) for r, c in zip(rows, cols)}
    mcc = float(corr[rows, cols].mean())
    return assign, mcc


def run_ivae(cfg, out_dir, epochs: int = 50) -> None:
    """Decoder-based Axis-A ICC via iVAE latent traversal against the frozen LSTM."""
    data = load_dataset(cfg.name, out_dir=out_dir)
    graph = data["graph"]
    k = graph.shape[0]
    X_train, X_test = data["X_train"], data["X_test"]

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"[07] no classifier at {ckpt}; run experiments/02_train_classifiers.py "
            f"--config {cfg.name} first."
        )
    clf = LSTMClassifier.load(ckpt)

    # Train iVAE with latent_dim = k so dims align one-to-one with channels.
    print(f"[07] training iVAE (latent_dim={k}, epochs={epochs}) on {X_train.shape[0]} seqs ...")
    ae = iVAE(latent_dim=k, n_segments=min(4, k), epochs=epochs,
              beta=0.3, kl_warmup_frac=0.3)
    ae.fit_unsupervised(X_train)

    # --- Sanity gates (report FIRST) ---
    Z = ae.encode(X_test)                         # (N, k)
    X_recon = ae.decode(Z)                         # (N, T, k)
    recon_mse = float(np.mean((X_recon - X_test) ** 2))
    f_x = np.asarray(clf.predict(X_test)).reshape(-1)
    f_recon = np.asarray(clf.predict(X_recon)).reshape(-1)
    recon_label_agreement = float(np.mean(f_recon == f_x))

    # Ground-truth factors = per-channel final values (the label-driving repr).
    F = X_test[:, -1, :]                           # (N, k)
    assign, mcc_val = _align_latents_to_channels(Z, F)

    relevant_channels = _ancestors_of(0, graph)    # label channel 0 + ancestors
    parent_dims = [i for i in range(k) if assign.get(i) in relevant_channels]
    nonparent_dims = [i for i in range(k) if assign.get(i) not in relevant_channels]

    interpretable = recon_label_agreement >= RECON_AGREEMENT_GATE
    print(f"[07] GATES: recon_mse={recon_mse:.4f}  recon_label_agreement={recon_label_agreement:.2f} "
          f"(gate>={RECON_AGREEMENT_GATE})  MCC(latent,channel)={mcc_val:.2f}")
    print(f"[07] label-relevant channels (anc. of 0): {sorted(relevant_channels)}; "
          f"latent->channel assign: {assign}")
    if not interpretable:
        print("[07] recon_label_agreement below gate -> ICC is NOT interpretable on this "
              "run (iVAE reconstruction does not preserve the classifier's decision). "
              "Reporting gates only; this is itself an honest Axis-A finding.")

    # --- ICC magnitude ladder (pre-registered c in {1,2,3}) ---
    rng = np.random.default_rng(cfg.seed)
    ladder = []
    for c in (1.0, 2.0, 3.0):
        icc = icc_latent(X_test, ae.encode, ae.decode, clf, delta=c,
                         scale_by_std=True, symmetric=True)  # (k,)
        parent_mean = float(np.mean([icc[i] for i in parent_dims])) if parent_dims else float("nan")
        nonparent_mean = float(np.mean([icc[i] for i in nonparent_dims])) if nonparent_dims else float("nan")
        # Permuted-assignment null: random partition of the same size as parent_dims.
        perm = rng.permutation(k)
        null_parent = perm[:len(parent_dims)]
        null_mean = float(np.mean([icc[i] for i in null_parent])) if len(null_parent) else float("nan")
        ladder.append({
            "c": c,
            "icc_per_dim": [float(v) for v in icc],
            "parent_aligned_mean": parent_mean,
            "nonparent_aligned_mean": nonparent_mean,
            "contrast": (parent_mean - nonparent_mean),
            "permuted_null_mean": null_mean,
        })
        print(f"[07] c={c:g}: ICC parent-aligned={parent_mean:.3f}  non-parent={nonparent_mean:.3f}  "
              f"contrast={parent_mean - nonparent_mean:+.3f}  (null={null_mean:.3f})")

    out = {
        "provenance": {
            "config": cfg.as_dict(), "method": "iVAE", "metric": "icc_latent",
            "n_eval": int(X_test.shape[0]), "latent_dim": k, "epochs": epochs,
            "note": "matched-baseline f(D(z)); +/-delta; delta_i=c*std(z_i); "
                    "latents aligned to channels via Hungarian on |corr|.",
        },
        "gates": {
            "recon_mse": recon_mse,
            "recon_label_agreement": recon_label_agreement,
            "interpretable": interpretable,
            "mcc_latent_channel": mcc_val,
        },
        "alignment": {
            "latent_to_channel": assign,
            "label_relevant_channels": sorted(int(c) for c in relevant_channels),
            "parent_aligned_dims": parent_dims,
            "nonparent_aligned_dims": nonparent_dims,
        },
        "icc_ladder": ladder,
    }
    res_dir = config_dir(cfg.name, "ivae")
    res_dir.mkdir(parents=True, exist_ok=True)
    dump_json(res_dir / "icc.json", out)
    print(f"[07] wrote {res_dir / 'icc.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    config_name: str,
    out_dir,
    method: str = "dynotears",
    n_cf: int = 40,
    epochs: int | None = None,
    intervention_prob: float = 0.3,
    sweep: bool = False,
    seed: int | None = None,
) -> None:
    """Dispatch to the auxiliary-method family selected by ``method``."""
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    n_epochs = epochs if epochs is not None else DEFAULT_EPOCHS.get(method, 80)

    if method in GRAPH_METHODS:
        run_graph_method(
            cfg, out_dir, method=method, n_cf=n_cf, epochs=n_epochs,
            intervention_prob=intervention_prob, sweep=sweep,
        )
    elif method == "ivae":
        run_ivae(cfg, out_dir, epochs=n_epochs)
    else:
        raise SystemExit(
            f"[07] unknown --method {method!r}; use one of "
            f"{', '.join((*GRAPH_METHODS, 'ivae'))}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 07: auxiliary causal methods -- self-graphing (DYNOTEARS / "
                    "CITRIS) with the Axis-B graph-error decomposition (H3), or "
                    "decoder-based Axis-A ICC (iVAE).",
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--method", default="dynotears", choices=(*GRAPH_METHODS, "ivae"),
        help="dynotears (default) -- the load-bearing graph-aware baseline; "
             "citris -- the honest secondary self-graphing method; "
             "ivae -- decoder-based Axis-A ICC.",
    )
    parser.add_argument("--n-cf", type=int, default=40,
                        help="instances for the decomposition (graph methods only)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="training epochs; defaults per method (citris 80, "
                             "ivae 50). Unused by dynotears.")
    parser.add_argument("--intervention-prob", type=float, default=0.3,
                        help="interventional-sequence rate (citris only)")
    parser.add_argument(
        "--sweep", action="store_true",
        help="also compute the graph-quality sweep (graph_error across a "
             "controlled true->random graph ladder + the method's real graph), "
             "demonstrating the decomposition's dynamic range (graph methods only).",
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
