"""Phase 07: auxiliary causal-method families outside the main 01-05 pipeline.

Phases 01-05 evaluate *counterfactual-generating* methods against a trained
LSTM. This phase covers the two method families that do not fit that shape --
they learn causal structure or a causal representation rather than producing a
counterfactual -- and scores each on the axis its output actually addresses.
Neither family is wired into the multi-seed orchestrator (Phase 08 ``seeds``),
and both are config-restricted; that is why they live here rather than in the
main pipeline.

Selected with ``--method``; each writes its own distinct report:

* ``dynotears`` / ``pcmciplus`` -> ``results/<config>/<method>/graph_error.json``
  **Self-graphing + Axis-A graph-error decomposition (H3).** Fits a
  self-graphing method on a *nonlinear* (MLP-mechanism) benchmark, reads off
  its inferred lag-1 causal graph, and reports:

  (a) Axis-A graph-recovery of the inferred graph vs. ground truth
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
  ~0.9) via continuous-optimization NOTEARS-style structure learning. This is
  the load-bearing graph-aware method that gives the Axis-A decomposition real
  dynamic range and H3 a genuine, non-circular positive. ``pcmciplus`` is
  PCMCIplus (Runge et al., 2020; `tigramite` package, installed as a normal
  PyPI dependency 2026-08-06, M4h) -- constraint-based/conditional-
  independence-testing causal discovery, structurally different from
  DYNOTEARS's continuous optimization. Also **observational**. Wired
  specifically as the second, different-model-class method
  ``docs/risk_register.md`` RISK-22 names for a DYNOTEARS-vs-PCMCIplus
  cross-method agreement check (``--cross-method-check``, below).

  Only nonlinear presets (``mechanism_type == "mlp"``, e.g. ``smoke_nl`` /
  ``full_nl``) are supported -- the decomposition needs an
  :class:`MLPMechanism` to build the inferred-graph rollout.

* ``cross_method_agreement`` -> ``results/<config>/cross_method_agreement/
  graph_agreement.json`` **Cross-method agreement check (M4h).** Fits both
  DYNOTEARS and PCMCIplus on the same data and reports the SHD between their
  independently inferred, density-matched graphs, plus each method's own
  AUC-vs-true-graph (Tier 1 synthetic data only -- no ground truth exists on
  real data, so the real-data case reports SHD-between-methods alone, labeled
  as structural agreement only, no correctness claim). Agreement alone is not
  reassuring if both methods share a bias; reporting both AUCs alongside the
  agreement number lets a reader tell "they agree and are both right" apart
  from "they agree and are both wrong."

**Removed 2026-08-03**: ``--method ivae``, the decoder-based
Axis-A ICC, went with ``causaltemp_xai/methods/concept/``. Concept-based methods
were descoped 2026-07-29 and no axis here scores them. ``metrics/axis_a.py``
itself is retained -- Axis A is paused, not disproven -- and stays covered by
``tests/test_axis_a_latent.py`` / ``tests/test_icc_latent.py``.

**Removed 2026-08-06** (M4h): ``--method citris`` and the
CITRIS-only ``--intervention-prob``/``--epochs`` flags. CITRIS (representation
learning; needed intervention-labeled data, never identified above chance at
smoke scale) is fully deleted, including its vendored submodule. PCMCIplus
replaces it as the second self-graphing method.

Usage
-----
    uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method dynotears
    uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method pcmciplus
    uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method cross_method_agreement

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual  # noqa: E402
from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.methods.causal import DYNOTEARS, PCMCIPlus  # noqa: E402
from causaltemp_xai.metrics.axis_a import compute_axis_a  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402
from causaltemp_xai.scm.intervention import (  # noqa: E402
    derive_intervention_t,
    is_vacuous_intervention,
)
from experiments._common import (  # noqa: E402
    ORACLE_SHIFT,
    build_masked_mechanism,
    build_oracle_interventions,
    config_dir,
    dump_json,
    select_flip_candidates,
    set_run_context,
)

GRAPH_METHODS = ("dynotears", "pcmciplus")


# ---------------------------------------------------------------------------
# Report 1: self-graphing + Axis-A graph-error decomposition (dynotears/pcmciplus)
# ---------------------------------------------------------------------------


def _mean_soft_cf_faith(X_sel, cfs, graph, mechanism, scorer) -> float:
    """Mean soft rollout CF-faith of ``cfs`` scored against ``(graph, mechanism)``.

    ``intervention_t`` is derived per instance from ``(x, x_cf)`` (mechanism-
    independent), so the true-graph and inferred-graph scorings use the same
    intervention point and differ only in the mechanism the CF is checked
    against.

    ``nanmean``, not ``mean`` (bug found 2026-08-04): ``CFfaith.score``
    deliberately returns NaN when ``intervention_t >= T - 1`` -- the
    degeneracy gate documented on that method, since a rollout window of
    length 0 has no evidence either way. A method whose derived intervention
    lands on the last step for even one instance (CftsCels does, on ~60% of
    `full_nl`'s selection) poisoned the *entire* method's mean under plain
    ``np.mean``, silently. Every other CF-faith aggregator in this codebase
    (``causaltemp_xai/eval.py``, ``experiments/_common.py``) already treats a
    gated instance as an abstention via ``nanmean``, not a NaN result.
    """
    vals = []
    for x, x_cf in zip(np.asarray(X_sel, dtype=float), np.asarray(cfs, dtype=float)):
        t = derive_intervention_t(x, x_cf)
        vals.append(scorer.score(x, x_cf, t, graph, mechanism)["soft"])
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


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
        method = cf_path.stem[len("X_cf_") :]
        cfs = np.load(cf_path)
        r_gt = _mean_soft_cf_faith(X_sel, cfs, graph, true_mech, rollout)
        p_gt = _mean_soft_cf_faith(X_sel, cfs, graph, true_mech, pearl)
        rows.append(
            {
                "method": method,
                "cf_faith_gt": r_gt,
                "propagation_error": float(1.0 - r_gt),
                "cf_faith_pearl_gt": p_gt,
                "propagation_error_pearl": float(1.0 - p_gt),
            }
        )
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


def _inferred_cf_faith(graph, mech, adj_pred, X_sel, oracle_ints, rollout) -> tuple[float, float]:
    """Mean soft rollout CF-faith of the oracle CF *derived from* ``adj_pred``
    (a masked mechanism restricted to the inferred edges), scored against the
    true mechanism, plus the fraction of those inferred CFs that are vacuous
    (RISK-17: a masked graph that makes the inferred intervention collapse to
    no-op is a different failure from one that mispropagates, and must not be
    invisible inside a low ``graph_error``). Lower CF-faith => more
    graph-induced divergence."""
    inf_mech = build_masked_mechanism(mech, adj_pred)
    vals = []
    n_vacuous = 0
    for (t0, node, _true_cf), x in zip(oracle_ints, X_sel):
        value = float(x[t0, node]) + ORACLE_SHIFT
        inf_cf = structural_counterfactual(x, inf_mech, t0, node, value, noiseless=True)
        vals.append(rollout.score(x, inf_cf, t0, graph, mech)["soft"])
        if is_vacuous_intervention(x, inf_cf, inf_mech, t0=t0):
            n_vacuous += 1
    return float(np.mean(vals)), float(n_vacuous) / len(oracle_ints)


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


def _graph_quality_sweep(
    graph,
    mech,
    X_sel,
    oracle_ints,
    rollout,
    cf_faith_gt,
    method_adj,
    method_auc,
    method_label,
    seed=0,
) -> list[dict]:
    """Graph-error across a controlled graph-quality ladder.

    Points: the true graph (0 by construction), progressively corrupted graphs
    (``frac`` of edges rewired), a fully random graph, and the method's
    *actual* recovered graph as the real-method anchor. Graph quality is
    reported as SHD-to-true and (for the corruption ladder) the corrupted
    fraction.

    **Measured outcome, `full_nl`, 3 seeds (2026-08-05) — read this before
    citing ``graph_error`` as a graph-quality measure.** This docstring
    previously asserted that ``graph_error`` "must span ~0 for a good graph up
    to large for a random graph". It does not, and the data wins (R5). The
    ladder is monotone in the mean but has almost no dynamic range: destroying
    the graph entirely (AUC 1.00 -> 0.39, SHD 0 -> 35) costs a mean
    ``graph_error`` of **0.0033** (per-seed 0.0018 / 0.0040 / 0.0039).

    The insensitivity is specific to the *graph* axis, not to the metric: on
    the same runs ``propagation_error`` spans 0.0000 (CARLA) to 0.1771
    (CftsCOMTE), i.e. **~54x more dynamic range across methods than across
    graph quality**. So the decomposition discriminates *methods* but not
    *graphs* here.

    **Dissipation hypothesis CONFIRMED (2026-08-05, `b56d0d8`).** The above
    explanation was "unverified" pending a non-dissipative test; it no longer
    is. Run on ``smoke_spring`` (M4c, rho ~ +0.02, vs. `full_nl`'s rho ~ -0.29),
    3 seeds: ``graph_error`` reaches 0.09-0.54 across the same corruption
    ladder that produced 0.0018-0.0040 on `full_nl`. **Caveat (2026-08-11):**
    the ratio is scale-confounded -- ``graph_error`` is in raw state units and
    sigma differs across families, so the headline "45x-123x" overstates it.
    Graph quality is near-irrelevant
    when effects attenuate before parent-set differences propagate
    (dissipative), and bites hard once effects persist (non-dissipative).
    ``graph_error`` is therefore a graph-quality measure **conditional on**
    the mechanism family's contraction rate, not a general one -- report it
    with that family's measured rho, not as a standalone number.
    """
    from causaltemp_xai.metrics.axis_a import graph_auc

    true_bin = (graph > 0).astype(int)
    rng = np.random.default_rng(seed)
    rows = []

    def _point(label, adj, auc):
        rows.append(
            _score_graph_point(
                graph, mech, X_sel, oracle_ints, rollout, cf_faith_gt, label, adj, auc
            )
        )

    # Controlled ladder: true -> increasingly corrupted -> random.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        adj = _corrupt_graph(true_bin, frac, rng)
        auc = graph_auc(true_bin, adj.astype(float)) if frac > 0 else 1.0
        _point(f"corrupt_frac={frac:g}", adj, auc)
    # Real-method anchor.
    _point(method_label, method_adj, method_auc)
    return rows


def _score_graph_point(
    graph, mech, X_sel, oracle_ints, rollout, cf_faith_gt, label, adj, auc
) -> dict:
    """One ``_graph_quality_sweep`` row: score a single candidate graph ``adj``
    (a corruption-ladder point, or a real/ensemble-member inferred graph)
    against the true mechanism.

    Extracted 2026-08-06 (M4f) from what was previously a
    closure (``_point``) defined inside :func:`_graph_quality_sweep`, so the
    new DYNOTEARS ensemble path (:func:`_graph_quality_sweep_ensemble`) can
    call it directly -- once per ensemble member -- without duplicating this
    scoring logic or depending on ``_graph_quality_sweep``'s internal closure.
    :func:`_graph_quality_sweep` itself is unchanged in behavior: it now calls
    this function instead of a local closure, but produces byte-identical
    output (regression-tested).
    """
    from causaltemp_xai.metrics.axis_a import shd

    true_bin = (graph > 0).astype(int)
    cf_inf, frac_vac = _inferred_cf_faith(graph, mech, adj, X_sel, oracle_ints, rollout)
    return {
        "label": label,
        "graph_auc": (None if auc is None else float(auc)),
        "shd": float(shd(true_bin, adj)),
        "cf_faith_inferred": cf_inf,
        "graph_error": float(cf_faith_gt - cf_inf),
        "frac_vacuous": frac_vac,
    }


def _fit_graph_method_ensemble(
    X_all, k, L, n_true, B, method: str = "dynotears", seed=0
) -> list[np.ndarray]:
    """Fit a `GRAPH_METHODS` member ``B`` times, each on an independent
    bootstrap resample of ``X_all``'s N-axis (M4f, 2026-08-06;
    generalized beyond DYNOTEARS-only 2026-08-06, M4h, to also accept
    ``pcmciplus``).

    Trajectories in ``X_all`` are i.i.d. draws from the SCM, so resampling
    *which* trajectories feed a given fit is the statistically valid
    bootstrap unit here -- never resample within one trajectory's own
    timesteps, which would break the lag structure both methods need (each
    consumes the panel per-sequence; nothing about that changes for this to
    work, since it is the caller's array being resampled, not the fitting
    logic).

    Returns ``B`` density-matched binary adjacency arrays (same convention as
    :func:`run_graph_method`'s single-fit ``adj_pred``), one per ensemble
    member.
    """
    from causaltemp_xai.stats import bootstrap_resample_indices

    idx = bootstrap_resample_indices(n=X_all.shape[0], n_boot=B, seed=seed)
    members = []
    for b in range(B):
        if method == "dynotears":
            model = DYNOTEARS(k=k, p=L).fit(X_all[idx[b]])
        elif method == "pcmciplus":
            model = PCMCIPlus(k=k, tau_max=L).fit(X_all[idx[b]])
        else:
            raise ValueError(f"_fit_graph_method_ensemble: unknown method {method!r}")
        _, scores = model.inferred_graph(max_lag=L)
        members.append(_density_matched_adjacency(scores, n_true))
    return members


def _graph_quality_sweep_ensemble(
    graph, mech, X_sel, oracle_ints, rollout, cf_faith_gt, ensemble_adjs, method_label, seed=0
) -> dict:
    """Ensemble extension of :func:`_graph_quality_sweep` (M4f,
    2026-08-06): scores each of the ``B`` ensemble-member
    graphs (produced by :func:`_fit_dynotears_ensemble`) as its own
    real-method anchor point, then reports the ensemble's **spread**
    alongside the point estimate -- never collapsed to one number, per this
    project's standing design stance against opaque composites
    (`docs/general_plan.md` §7, RISK-12/17/18/20).

    The corruption ladder itself is *not* recomputed here: it corrupts the
    true graph, which does not depend on DYNOTEARS at all, so it is identical
    across every ensemble member and is computed exactly once by the existing
    :func:`_graph_quality_sweep` call this function's caller also makes.

    Returns
    -------
    dict with:
        ``ensemble_anchor_points``: one :func:`_score_graph_point` row per
            ensemble member.
        ``graph_error_ensemble``: ``{mean, ci_lo, ci_hi, n}`` -- a percentile
            bootstrap CI (:func:`causaltemp_xai.stats.bootstrap_ci`) treating
            the ``B`` ensemble members' ``graph_error`` values themselves as
            the i.i.d. sample being resampled (legitimate: they are ``B``
            independent draws). Collapses to an exact point when every member
            agrees (including the ``B`` identical-resample degenerate case).
        ``mean_pairwise_shd``: mean SHD across all ``C(B,2)`` pairs of the
            ensemble's *own* inferred graphs (not against the true graph) --
            the ground-truth-free inter-graph disagreement reading.
        ``ensemble_b``: ``B``.
    """
    from causaltemp_xai.metrics.axis_a import graph_auc, shd
    from causaltemp_xai.stats import bootstrap_ci

    true_bin = (graph > 0).astype(int)
    anchor_points = [
        _score_graph_point(
            graph,
            mech,
            X_sel,
            oracle_ints,
            rollout,
            cf_faith_gt,
            f"{method_label}_b{b}",
            adj,
            graph_auc(true_bin, adj.astype(float)),
        )
        for b, adj in enumerate(ensemble_adjs)
    ]

    graph_errors = [row["graph_error"] for row in anchor_points]
    ci = bootstrap_ci(graph_errors, seed=seed)

    B = len(ensemble_adjs)
    pairwise_shd = [
        shd(ensemble_adjs[i], ensemble_adjs[j]) for i in range(B) for j in range(i + 1, B)
    ]
    mean_pairwise_shd = float(np.mean(pairwise_shd)) if pairwise_shd else 0.0

    return {
        "ensemble_anchor_points": anchor_points,
        "graph_error_ensemble": {"mean": ci.mean, "ci_lo": ci.ci_lo, "ci_hi": ci.ci_hi, "n": ci.n},
        "mean_pairwise_shd": mean_pairwise_shd,
        "ensemble_b": B,
    }


def run_cross_method_agreement(cfg, out_dir, pc_alpha: float = 0.05) -> None:
    """DYNOTEARS-vs-PCMCIplus cross-method agreement check (M4h, 2026-08-06,
    -- the concrete RISK-22 mitigation.

    Works on **any** Tier-1 synthetic config, including `linear` (VAR)
    -- unlike `run_graph_method`'s CF-faith decomposition (restricted to
    `mlp`/`spring`, since it calls `build_masked_mechanism`, which
    `LinearMechanism` genuinely cannot support), this function never masks a
    mechanism -- it only fits both methods and compares raw adjacencies, which
    is mechanism-type-agnostic. Guard dropped 2026-08-06 (M4i)
    after confirming it was never load-bearing here.

    **Reference-graph caveat for ``spring`` (P0-4, 2026-08-11).**
    ``SpringMechanism``'s ``graph`` deliberately excludes the position<-velocity
    coupling every particle's own dynamics depend on -- see the class
    docstring's "not graph-worthy" convention, the same one ``MLPMechanism``
    uses for its ``decay_i`` self-term. That coupling is the strongest single
    dependency in the system (measured: 3x any true cross-particle edge). A
    discovery method that recovers it is recovering a real, dominant
    dependency and is scored *wrong* for doing so, because ``true_bin`` here
    omits it. A below-chance ``dynotears_auc``/``pcmciplus_auc`` on ``spring``
    is therefore not necessarily evidence the method failed to find structure
    -- it may be evidence the method found structure this function's reference
    graph does not credit. Read spring's AUC alongside its ``shd_between_methods``
    (structural agreement, which needs no reference) before concluding either
    method is worse than the other on this family.

    Fits both methods independently on the same observational data, density-
    matches each to the true edge count, and reports:

    - ``shd_between_methods``: SHD between the two methods' own inferred
      graphs (needs no ground truth -- the pure structural-agreement number,
      the only one available on real data).
    - ``dynotears_auc`` / ``pcmciplus_auc``: each method's own AUC against
      the *true* graph (Tier 1 synthetic data only, since ground truth exists
      here).
    - ``dynotears_shd_to_true`` / ``pcmciplus_shd_to_true``: same idea, SHD
      form.

    **Agreement alone is not reassuring.** If DYNOTEARS and PCMCIplus shared
    the same bias, a low ``shd_between_methods`` would look reassuring while
    both were confidently wrong in the same way -- exactly the failure mode
    within-model bootstrap resampling (M4f) cannot detect (`docs/
    risk_register.md` RISK-22). Reporting each method's own AUC-vs-true-graph
    alongside the agreement number lets a reader tell "they agree and are
    both right" apart from "they agree and are both wrong." Two structurally
    different model classes (DYNOTEARS's continuous optimisation vs.
    PCMCIplus's conditional-independence testing) agreeing is evidence
    against shared bias, not proof against it.
    """
    from causaltemp_xai.metrics.axis_a import graph_auc, shd

    data = load_dataset(cfg.name, out_dir=out_dir)
    graph = data["graph"]
    k, _, L = graph.shape
    X_all = data["X_train"]
    true_bin = (graph > 0).astype(int)
    n_true = int(true_bin.sum())

    print(f"[07] cross-method agreement: fitting DYNOTEARS on {X_all.shape[0]} sequences ...")
    dyn_model = DYNOTEARS(k=k, p=L).fit(X_all)
    _, dyn_scores = dyn_model.inferred_graph(max_lag=L)
    dyn_adj = _density_matched_adjacency(dyn_scores, n_true)

    print(f"[07] cross-method agreement: fitting PCMCIplus on {X_all.shape[0]} sequences ...")
    pcmci_model = PCMCIPlus(k=k, tau_max=L, pc_alpha=pc_alpha).fit(X_all)
    _, pcmci_scores = pcmci_model.inferred_graph(max_lag=L)
    pcmci_adj = _density_matched_adjacency(pcmci_scores, n_true)

    out = {
        "provenance": {"config": cfg.as_dict(), "pc_alpha": pc_alpha},
        "shd_between_methods": float(shd(dyn_adj, pcmci_adj)),
        "dynotears_auc": float(graph_auc(true_bin, dyn_scores)),
        "pcmciplus_auc": float(graph_auc(true_bin, pcmci_scores)),
        "dynotears_shd_to_true": float(shd(true_bin, dyn_adj)),
        "pcmciplus_shd_to_true": float(shd(true_bin, pcmci_adj)),
        "note": (
            "shd_between_methods (structural agreement) does not by itself imply "
            "correctness -- see dynotears_auc/pcmciplus_auc for each method's own "
            "accuracy against the true graph. RISK-22 (docs/risk_register.md): "
            "agreement between two differently-biased model classes is evidence "
            "against shared bias, not proof against it."
        ),
    }
    out_dir_res = config_dir(cfg.name, "cross_method_agreement")
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_agreement.json", out)

    print(
        f"[07] cross-method agreement: SHD(dynotears,pcmciplus)="
        f"{out['shd_between_methods']:.0f} dynotears_AUC={out['dynotears_auc']:.3f} "
        f"pcmciplus_AUC={out['pcmciplus_auc']:.3f}"
    )
    print(f"[07] wrote {out_dir_res / 'graph_agreement.json'}")


def run_graph_method(
    cfg,
    out_dir,
    method: str,
    n_cf: int = 40,
    sweep: bool = False,
    ensemble_b: int = 0,
    pc_alpha: float = 0.05,
) -> None:
    """Self-graphing + Axis-A graph-error decomposition for ``dynotears``/``pcmciplus``."""
    # Families whose mechanism `build_masked_mechanism` can restrict to an
    # inferred graph. Spring was added 2026-08-05 specifically to test the
    # dissipation hypothesis for M4e's near-flat graph-quality ladder: the
    # ladder needs a NON-dissipative family (rho ~ 1) as its comparison arm.
    # (Kuramoto was the other such family; removed 2026-08-11.)
    _MASKABLE = ("mlp", "spring")
    if cfg.mechanism_type not in _MASKABLE:
        raise SystemExit(
            f"[07] self-graphing graph-error needs a config whose mechanism can be "
            f"restricted to an inferred graph {_MASKABLE}; {cfg.name!r} is "
            f"mechanism_type={cfg.mechanism_type!r}. Try --config smoke_nl."
        )
    if ensemble_b > 0 and not sweep:
        raise SystemExit(
            "[07] --ensemble-b requires --sweep (M4f): the ensemble's spread is only "
            "meaningful alongside the graph-quality sweep it extends."
        )

    data = load_dataset(cfg.name, out_dir=out_dir)
    graph, mech = data["graph"], data["mechanism"]
    k, _, L = graph.shape

    # 1. Fit the self-graphing method and read off its inferred lag-1 graph.
    #    DYNOTEARS (default) is a classical temporal causal-discovery baseline
    #    that recovers the benchmark's near-linear structure from OBSERVATIONAL
    #    data via continuous-optimization NOTEARS-style structure learning --
    #    the load-bearing graph-aware method for Axis A / H3. PCMCIplus (M4h,
    #    2026-08-06) is constraint-based/conditional-independence-testing
    #    discovery, structurally different from DYNOTEARS, wired specifically
    #    for the cross-method agreement check (RISK-22).
    X_all = data["X_train"]
    pcmciplus_meta = None
    if method == "dynotears":
        print(f"[07] fitting DYNOTEARS on {X_all.shape[0]} sequences (k={k}, p={L}) ...")
        model = DYNOTEARS(k=k, p=L).fit(X_all)
        _, scores = model.inferred_graph(max_lag=L)
    elif method == "pcmciplus":
        print(
            f"[07] fitting PCMCIplus on {X_all.shape[0]} sequences "
            f"(k={k}, tau_max={L}, pc_alpha={pc_alpha}) ..."
        )
        model = PCMCIPlus(k=k, tau_max=L, pc_alpha=pc_alpha).fit(X_all)
        _, scores = model.inferred_graph(max_lag=L)
        pcmciplus_meta = {
            "source": "tigramite (github.com/jakobrunge/tigramite), PyPI dependency",
            "tau_max": L,
            "pc_alpha": pc_alpha,
        }
    else:
        raise SystemExit(f"[07] unknown graph method {method!r}; use one of {GRAPH_METHODS}")

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

    # 3. Axis A (graph recovery + oracle decomposition) in one call. The oracle
    #    row is the perfect-propagator reference: propagation_error == 0, so its
    #    graph_error is the pure cost of the inferred graph.
    axis_a = compute_axis_a(
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
    #    to large (random graph), so the Axis-A decomposition is demonstrably
    #    discriminative rather than evaluated at a single near-perfect point.
    sweep_rows = None
    if sweep:
        sweep_rows = _graph_quality_sweep(
            graph,
            mech,
            X_sel,
            oracle_ints,
            rollout,
            cf_faith_gt,
            method_adj=adj_pred,
            method_auc=axis_a.get("AUC"),
            method_label=method,
            seed=cfg.seed,
        )

    # 6. Uncertainty-aware ensemble (M4f, optional): B independent DYNOTEARS
    #    refits, each on a bootstrap resample of X_all's N-axis, scored the
    #    same way as the real-method anchor above, with the ensemble's spread
    #    reported alongside -- see _graph_quality_sweep_ensemble's docstring.
    #    Additive: the plain single-fit "graph_quality_sweep" key above is
    #    always still written, so nothing reading it breaks.
    ensemble_result = None
    if ensemble_b > 0:
        print(f"[07] fitting {method} ensemble (B={ensemble_b}) ...")
        ensemble_adjs = _fit_graph_method_ensemble(
            X_all, k, L, n_true, ensemble_b, method=method, seed=cfg.seed
        )
        ensemble_result = _graph_quality_sweep_ensemble(
            graph,
            mech,
            X_sel,
            oracle_ints,
            rollout,
            cf_faith_gt,
            ensemble_adjs,
            method_label=method,
            seed=cfg.seed,
        )

    out = {
        "provenance": {
            "config": cfg.as_dict(),
            "method": method,
            "n_cf": len(X_sel),
            "dynotears": (
                {
                    "source": "third_party/causalnex_repo (github.com/mckinsey/causalnex)",
                    "lambda_a": model.lambda_a,
                    "lambda_w": model.lambda_w,
                    "p": model.p,
                }
                if method == "dynotears"
                else None
            ),
            "pcmciplus": pcmciplus_meta,
        },
        "axis_a": axis_a,
        "oracle_decomposition": {
            "cf_faith_gt": axis_a["cf_faith_gt"],
            "cf_faith_inferred": axis_a["cf_faith_inferred"],
            "graph_error": axis_a["graph_error"],
            "propagation_error": axis_a["propagation_error"],
        },
        "methods": method_rows,
        "graph_quality_sweep": sweep_rows,
        "graph_quality_sweep_ensemble": ensemble_result,
        "inferred_edges": int(adj_pred.sum()),
        "true_edges": int((graph > 0).sum()),
    }

    out_dir_res = config_dir(cfg.name, method)
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_error.json", out)

    print(
        f"[07] {method} Axis A: SHD={axis_a['SHD']:.0f} LagAcc={axis_a['LagAcc']:.2f} "
        f"AUC={axis_a.get('AUC', float('nan')):.3f}"
    )
    print(
        f"[07] oracle (perfect propagator): graph_error={axis_a['graph_error']:+.3f} "
        f"propagation_error={axis_a['propagation_error']:+.3f}"
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
        print(
            "[07] no persisted CF methods found (run experiments/03 first) -- "
            "per-method decomposition skipped"
        )
    if sweep_rows:
        print("[07] graph-quality sweep (graph_error vs graph quality):")
        print(f"       {'graph':<18} {'AUC':>6} {'SHD':>5} {'graph_error':>12}")
        for r in sweep_rows:
            auc = "  n/a" if r["graph_auc"] is None else f"{r['graph_auc']:.2f}"
            print(
                f"       {r['label']:<18} {auc:>6} {r['shd']:>5.0f} " f"{r['graph_error']:>+12.3f}"
            )
    print(f"[07] wrote {out_dir_res / 'graph_error.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _generate_pn_cfs(cfg, out_dir, clf, data, n_cf: int, methods_filter=None):
    """Generate (and cache) the **necessity**-direction counterfactuals.

    The PS direction reuses Phase 03's arrays, but PN needs the mirror image —
    instances the classifier already puts in the *target* class, with CFs
    aimed at leaving it — which Phase 03 never generates. Rather than
    duplicating the method registry, this reuses Phase 03's own
    ``build_methods``/``generate_cfs`` via ``importlib`` (the same pattern
    Phase 08 uses to drive phases 01-04), constructed with ``target_class=0``.

    Arrays are cached under ``cf_pn/`` and reused on re-run: CF generation is
    the expensive part of the pipeline, and PN doubles it.
    """
    import importlib

    phase03 = importlib.import_module("experiments.03_run_cf_methods")

    res_dir = config_dir(cfg.name, "lstm")
    pn_dir = res_dir / "cf_pn"
    pn_dir.mkdir(parents=True, exist_ok=True)

    x_sel_path = pn_dir / "X_sel.npy"
    if x_sel_path.exists():
        X_sel = np.load(x_sel_path)
    else:
        # from_class=1: already in the target class -- the population PN
        # conditions on (see select_flip_candidates / pns_metric_design.md).
        idx = select_flip_candidates(clf, data["X_test"], n_cf, target_class=1, from_class=1)
        X_sel = data["X_test"][idx]
        np.save(x_sel_path, X_sel)

    methods = phase03.build_methods(data["X_train"], data["Y_train"], target_class=0)
    if methods_filter:
        methods = {k: v for k, v in methods.items() if k in methods_filter}

    cfs = {}
    for name, method in methods.items():
        dest = pn_dir / f"X_cf_{name}.npy"
        if dest.exists():
            cfs[name] = np.load(dest)
            continue
        print(f"[07]   generating PN-direction CFs: {name} ...")
        try:
            arr = phase03.generate_cfs(method, X_sel, clf, data["graph"], data["mechanism"])
            np.save(dest, arr)
            cfs[name] = arr
        except Exception as exc:
            print(f"[07]   {name} FAILED: {exc}")
    return X_sel, cfs


def run_pns(
    cfg,
    out_dir,
    n_cf: int = 40,
    with_pn: bool = False,
    methods_filter=None,
    schedule: bool = False,
) -> None:
    """Necessity/sufficiency gap: the model's causal claim vs the world's.

    The **PS** direction reuses the counterfactual arrays Phase 03 already
    wrote — nothing is regenerated. The **PN** direction (``--with-pn``) needs
    counterfactuals Phase 03 never produces (target class -> non-target), so
    they are generated here and cached under ``cf_pn/``.

    Without ``--with-pn`` this reports PS only and writes ``PN_world: null``;
    a combined PNS is **never** synthesised from a missing term (R3 — on
    flip-candidates alone the estimand is sufficiency, not PNS).

    ``schedule=True`` audits each CF against the **whole** multi-timestep
    intervention it implies rather than the single ``do()`` at ``t0``
    (RISK-18), and writes to ``pns_schedule.json`` so the default-mode file —
    which every committed number was produced with — is never clobbered.
    ``do_complexity`` is reported in both modes.

    Writes ``pns.json`` under the config's classifier directory.
    """
    from causaltemp_xai.metrics.pns import (
        pns_direction,
        pns_from_directions,
        recover_label_threshold,
        scm_label,
    )

    data = load_dataset(cfg.name, out_dir=out_dir)
    mech = data["mechanism"]
    X_all = np.concatenate([data[f"X_{s}"] for s in ("train", "val", "test")])
    Y_all = np.concatenate([data[f"Y_{s}"] for s in ("train", "val", "test")])
    label = cfg.label_functional()
    theta = recover_label_threshold(X_all, Y_all, label_fn=label)

    res_dir = config_dir(cfg.name, "lstm")
    cf_dir = res_dir / "cf"
    x_sel_path = cf_dir / "X_sel.npy"
    if not x_sel_path.exists():
        raise SystemExit(f"[07] no {x_sel_path}; run experiments/03_run_cf_methods.py first")
    X_sel = np.load(x_sel_path)
    clf = LSTMClassifier.load(Path(out_dir) / cfg.name / "lstm.pt")

    fmt = lambda v: "nan" if v != v else f"{v:+.2f}"  # noqa: E731

    def _report(label, rows):
        print(f"[07] {label}")
        for name, out in rows.items():
            print(
                f"     {name:14s} A={fmt(out['A_model_proposed'])} "
                f"B={fmt(out['B_model_oracle'])} C={fmt(out['C_world_oracle'])} | "
                f"d_total={fmt(out['delta_total'])} d_traj={fmt(out['delta_trajectory'])} "
                f"d_out={fmt(out['delta_outcome'])} "
                f"D={out['do_complexity_mean']:.1f} (n={out['n_scorable']}/{out['n']})"
            )

    T = X_all.shape[1]
    print(
        f"[07] PNS on '{cfg.name}' -- theta={theta:+.6f} "
        f"label={cfg.label_fn}@t={label.label_site(T)} of T={T} "
        f"mode={'schedule' if schedule else 'single-slice'}"
    )
    ps_rows = {}
    for cf_path in sorted(cf_dir.glob("X_cf_*.npy")):
        name = cf_path.stem[len("X_cf_") :]
        ps_rows[name] = pns_direction(
            X_sel,
            np.load(cf_path),
            clf,
            mech,
            theta,
            target_class=1,
            schedule=schedule,
            label_fn=label,
        )
    _report("PS direction (sufficiency): non-target -> target", ps_rows)

    payload = {
        "seed": cfg.seed,
        "config": cfg.name,
        "label_threshold": theta,
        "label_fn": cfg.label_fn,
        "label_site": label.label_site(T),
        "T": int(T),
        "schedule_mode": bool(schedule),
        "PS": ps_rows,
    }

    if not with_pn:
        payload.update(
            direction="PS",
            PN_world=None,
            PN_note=(
                "PN direction not run (pass --with-pn). Combined PNS is "
                "deliberately not synthesised from a missing term."
            ),
        )
    else:
        X_sel_pn, pn_cfs = _generate_pn_cfs(cfg, out_dir, clf, data, n_cf, methods_filter)
        pn_rows = {
            name: pns_direction(
                X_sel_pn,
                arr,
                clf,
                mech,
                theta,
                target_class=0,
                schedule=schedule,
                label_fn=label,
            )
            for name, arr in sorted(pn_cfs.items())
        }
        _report("PN direction (necessity): target -> non-target", pn_rows)

        # Population weights for PNS = P(x,y)*PN + P(x',y')*PS, taken from the
        # *world's* labels (this is a world-side quantity, so the classifier's
        # opinion of the class balance is not the right weight).
        world_y = np.array([scm_label(x, theta, label_fn=label) for x in X_all])
        p_xy = float((world_y == 1).mean())
        p_xpyp = float(1.0 - p_xy)

        combined = {}
        for name in sorted(set(ps_rows) & set(pn_rows)):
            combined[name] = pns_from_directions(ps_rows[name], pn_rows[name], p_xy, p_xpyp)
        payload.update(direction="PNS", PN=pn_rows, PNS=combined, p_xy=p_xy, p_xpyp=p_xpyp)

        print(f"[07] PNS combined  (P(x,y)={p_xy:.2f}, P(x',y')={p_xpyp:.2f})")
        for name, c in combined.items():
            print(
                f"     {name:14s} PNS={fmt(c['PNS_world'])} "
                f"= {p_xy:.2f}*PN({fmt(c['PN_world'])}) + {p_xpyp:.2f}*PS({fmt(c['PS_world'])})"
                f"   | d_total PS={fmt(c['PS_delta_total'])} PN={fmt(c['PN_delta_total'])}"
            )

    fname = "pns_schedule.json" if schedule else "pns.json"
    dump_json(res_dir / fname, payload)
    print(f"[07] wrote {res_dir / fname}")


def run(
    config_name: str,
    out_dir,
    method: str = "dynotears",
    n_cf: int = 40,
    sweep: bool = False,
    seed: int | None = None,
    with_pn: bool = False,
    methods_filter=None,
    schedule: bool = False,
    ensemble_b: int = 0,
    pc_alpha: float = 0.05,
) -> None:
    """Dispatch to the auxiliary-method family selected by ``method``."""
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    set_run_context(seed=cfg.seed, config=cfg.name)

    if method in GRAPH_METHODS:
        run_graph_method(
            cfg,
            out_dir,
            method=method,
            n_cf=n_cf,
            sweep=sweep,
            ensemble_b=ensemble_b,
            pc_alpha=pc_alpha,
        )
    elif method == "cross_method_agreement":
        run_cross_method_agreement(cfg, out_dir, pc_alpha=pc_alpha)
    elif method == "pns":
        run_pns(
            cfg,
            out_dir,
            n_cf=n_cf,
            with_pn=with_pn,
            methods_filter=methods_filter,
            schedule=schedule,
        )
    else:
        raise SystemExit(
            f"[07] unknown --method {method!r}; use one of "
            f"{', '.join((*GRAPH_METHODS, 'cross_method_agreement', 'pns'))}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 07: auxiliary causal methods -- self-graphing (DYNOTEARS / "
        "PCMCIplus) with the Axis-A graph-error decomposition (H3), a "
        "DYNOTEARS-vs-PCMCIplus cross-method agreement check, or "
        "the necessity/sufficiency audit (pns).",
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--method",
        default="dynotears",
        choices=(*GRAPH_METHODS, "cross_method_agreement", "pns"),
        help="dynotears (default) -- the load-bearing graph-aware baseline; "
        "pcmciplus -- constraint-based second self-graphing method (M4h); "
        "cross_method_agreement -- fits both and reports their SHD + AUCs (RISK-22); "
        "pns -- necessity/sufficiency gap (model vs world).",
    )
    parser.add_argument(
        "--n-cf", type=int, default=40, help="instances for the decomposition (graph methods only)"
    )
    parser.add_argument(
        "--pc-alpha",
        type=float,
        default=0.05,
        help="pcmciplus / cross_method_agreement only: PCMCIplus's PC-stage "
        "significance threshold.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="pns --with-pn only: subset of CF methods to generate the "
        "necessity direction for (defaults to all registered).",
    )
    parser.add_argument(
        "--with-pn",
        action="store_true",
        help="pns only: also generate the necessity-direction CFs (target -> "
        "non-target) and report genuine PNS. Doubles CF generation; arrays are "
        "cached under cf_pn/ and reused.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="pns only: audit each CF against the whole multi-timestep "
        "intervention it implies rather than the single do() at t0 (RISK-18). "
        "Writes pns_schedule.json; the default-mode file is left alone.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also compute the graph-quality sweep (graph_error across a "
        "controlled true->random graph ladder + the method's real graph), "
        "demonstrating the decomposition's dynamic range (graph methods only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Multi-seed replicate (M2): override the config seed via seeded_variant.",
    )
    parser.add_argument(
        "--ensemble-b",
        type=int,
        default=0,
        help="dynotears --sweep only (M4f): fit B independent DYNOTEARS models, each on "
        "an N-axis bootstrap resample of the observational dataset, and report the "
        "ensemble's spread (graph_error CI, mean pairwise SHD) alongside the existing "
        "single-fit point estimate. 0 (default) = off, byte-identical to current behavior.",
    )
    args = parser.parse_args(argv)
    run(
        args.config,
        args.out_dir,
        method=args.method,
        n_cf=args.n_cf,
        sweep=args.sweep,
        seed=args.seed,
        with_pn=args.with_pn,
        methods_filter=args.methods,
        schedule=args.schedule,
        ensemble_b=args.ensemble_b,
        pc_alpha=args.pc_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
