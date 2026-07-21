"""Shared helpers for the numbered experiment-phase scripts (01-07).

Every phase script writes under a single ``results/`` tree so later phases
(and ``06_aggregate_and_report.py``) can discover what earlier phases produced purely
by path convention::

    results/<config_name>/<classifier>/cf/X_sel.npy
    results/<config_name>/<classifier>/cf/X_cf_<Method>.npy
    results/<config_name>/<classifier>/eval_<Method>.json
    results/<config_name>/<classifier>/per_instance.csv
    results/<config_name>/<classifier>/attribution.json
    results/<config_name>/<classifier>/axis_a_attribution.json
    results/<config_name>/<classifier>/shift_vr.json
    results/<config_name>/axis_b_benchmark.json      # dataset-level, no classifier needed
    results/<config_name>/oracle/...                 # nonlinear configs
    results/tables/table_axis_c_cf_faith.csv         # accumulated across runs
    results/figures/*.png

Axis routing
------------
The benchmark's four metric axes (``causaltemp_xai/metrics/axis_{a,b,c,d}.py``)
are evaluated on whichever kind of method they are actually suited to:

* **Axis C** (validity, proximity, sparsity, OOD, TRSI) + **CF-faith** —
  every CF-*generating* method (Wachter, CARLA, cfts-*, OracleCF-*). This is
  "does the counterfactual itself look good and respect the SCM." Axis C scores
  the CF as an artifact; CF-faith scores it against the mechanism (including
  the retroactive-edit gate).
* **Axis D** (Shift-VR + attribution input-sensitivity) — Shift-VR applies to
  CF methods (validity retention under a noise-distribution shift, computed
  live in Phase 03); input-sensitivity applies to attribution methods
  (stability of the saliency map under small input perturbations).
* **Axis A** (ICC attribution-mass, causal-coverage) — attribution/saliency
  methods (Integrated Gradients), scored against **ground-truth oracle
  interventions** built from the known SCM (:func:`build_oracle_interventions`)
  so ``int_channel`` / ``causal_parents`` are real, not proxies.
* **Axis B** (SHD, LagAcc, TV-Confounding) — no graph-*discovery* method
  exists in this pipeline, so Axis B is not a per-method score here. It is
  computed once per **dataset** (Phase 01) as a structural diagnostic of the
  benchmark's own causal graph (:func:`axis_b_benchmark_diagnostic`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TARGET_CLASS = 1


def config_dir(config_name: str, classifier: str = "lstm") -> Path:
    """``results/<config_name>/<classifier>/`` — created on demand."""
    d = RESULTS_DIR / config_name / classifier
    d.mkdir(parents=True, exist_ok=True)
    return d


def select_flip_candidates(clf, X_test, n_cf, target_class=TARGET_CLASS):
    """Indices of test instances the classifier predicts as NOT target_class."""
    preds = clf.predict(X_test)
    src = [i for i in range(len(X_test)) if preds[i] != target_class]
    if len(src) < n_cf:
        src = list(range(len(X_test)))
    return np.asarray(src[:n_cf], dtype=int)


def per_instance_records(benchmark, classifier, method_name, X_sel, CFs, graph, mech,
                          preds=None):
    """One record per (method, instance) with per-CF Axis-C + CF-faith metrics.

    Axis C's ``TRSI`` (mechanism-free temporal smoothness of the edit — a
    proxy, not a faithfulness criterion; see :func:`~causaltemp_xai.metrics.
    axis_c.trsi`) is included alongside validity/proximity/sparsity — the full
    Axis-C suite for a CF-*generating* method, per-instance using its own
    derived intervention timestep.

    Sparsity is reported three ways: the flat ``sparsity`` over all ``T×k``
    features, plus ``sparsity_channels`` / ``sparsity_timepoints`` — the
    fraction of channels (resp. timesteps) left *entirely* untouched. The flat
    score cannot distinguish "one variable, always" from "all variables, one
    moment"; the structured pair can.

    Also includes the **joint faithfulness-validity** columns
    ``cf_faith_{rollout,pearl}_hard_valid`` = hard-faith AND classifier-valid
    per instance (anti-gameability criterion, M1 2026-07-07): a tiny-edit CF
    can pass the faithfulness check without consulting the SCM, but earns no
    joint credit unless it also flips the classifier. Blank when ``preds`` is
    None (classifier-free oracle rows in Phase 05).
    """
    from causaltemp_xai.metrics.axis_c import proximity, sparsity, trsi
    from causaltemp_xai.metrics.cf_faith import CFfaith
    from causaltemp_xai.scm.intervention import derive_intervention_t

    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mech)
        p = pearl.score(x, x_cf, t, graph, mech)
        _spars_detail = sparsity(x, x_cf, return_detailed=True)
        valid_i = int(preds[i] == TARGET_CLASS) if preds is not None else None
        rows.append({
            "benchmark": benchmark,
            "classifier": classifier,
            "method": method_name,
            "instance": int(i),
            "validity": (valid_i if valid_i is not None else ""),
            "proximity_l1": proximity(x, x_cf, norm="l1"),
            "proximity_l2": proximity(x, x_cf, norm="l2"),
            "sparsity": sparsity(x, x_cf),
            "sparsity_channels": _spars_detail["channels"],
            "sparsity_timepoints": _spars_detail["timepoints"],
            "trsi": trsi(x_cf, x),
            "intervention_t": int(t),
            "cf_faith_rollout_hard": r["hard"],
            "cf_faith_rollout_soft": r["soft"],
            "cf_faith_pearl_hard": p["hard"],
            "cf_faith_pearl_soft": p["soft"],
            # Joint faithfulness-validity (anti-gameability, M1 2026-07-07).
            "cf_faith_rollout_hard_valid": (r["hard"] * valid_i if valid_i is not None else ""),
            "cf_faith_pearl_hard_valid": (p["hard"] * valid_i if valid_i is not None else ""),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Overwrite ``path`` with ``rows`` (first row's keys define the header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def append_table(path: Path, rows: list[dict]) -> None:
    """Append ``rows`` to an accumulating table CSV under ``results/tables/``.

    Rows accumulate across runs and across the pipeline's lifetime, so the
    on-disk header may predate the current metric schema. This function
    **reconciles** the two rather than assuming they match.

    Schema-drift guard (2026-07-15). Previously the header was written only when
    the file did not exist, and appends trusted that every run produced the same
    columns forever. When the Axis-C schema changed (``ivr`` removed, then
    ``sparsity_channels`` / ``sparsity_timepoints`` added) that assumption broke
    **silently**: rows with 15 fields were appended under a stale 16-field
    header, so any ``csv``/``pandas`` read shifted every column after ``trsi``
    left by one — for exactly the rows carrying the newest numbers, while the
    file still parsed without error. Silent misalignment of a results table is
    the worst failure mode available here, so on any drift we now migrate the
    file to the union schema (old rows get ``""`` for new columns) and say so on
    stdout. Column order follows the incoming rows, with any columns only
    present on disk appended after.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())

    if not path.exists():
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        return

    with open(path, newline="") as fh:
        existing = list(csv.DictReader(fh))
        old_cols = list(existing[0].keys()) if existing else cols

    if old_cols == cols:
        with open(path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols).writerows(rows)
        return

    # Drift: rewrite the whole table under the union schema.
    union = cols + [c for c in old_cols if c not in cols]
    dropped = [c for c in old_cols if c not in cols]
    added = [c for c in cols if c not in old_cols]
    print(
        f"[tables] schema drift in {path.name}: "
        f"+{added or 'none'} -{dropped or 'none'} -- migrating "
        f"{len(existing)} existing row(s) to the union schema"
    )
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=union, restval="")
        writer.writeheader()
        for r in existing:
            writer.writerow({c: r.get(c, "") for c in union})
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in union})


def aggregate_method_row(benchmark, classifier, method_name, instance_rows) -> dict:
    """Collapse per-instance rows for one method into a single mean-summary row."""
    def _mean(key):
        vals = [r[key] for r in instance_rows if r[key] not in (None, "")]
        return float(np.mean(vals)) if vals else None

    return {
        "benchmark": benchmark,
        "classifier": classifier,
        "method": method_name,
        "n": len(instance_rows),
        "validity": _mean("validity"),
        "proximity_l1": _mean("proximity_l1"),
        "proximity_l2": _mean("proximity_l2"),
        "sparsity": _mean("sparsity"),
        "sparsity_channels": _mean("sparsity_channels"),
        "sparsity_timepoints": _mean("sparsity_timepoints"),
        "trsi": _mean("trsi"),
        "cf_faith_rollout_hard": _mean("cf_faith_rollout_hard"),
        "cf_faith_rollout_soft": _mean("cf_faith_rollout_soft"),
        "cf_faith_pearl_hard": _mean("cf_faith_pearl_hard"),
        "cf_faith_pearl_soft": _mean("cf_faith_pearl_soft"),
        # Joint faithfulness-validity (None for classifier-free oracle rows).
        "cf_faith_rollout_hard_valid": _mean("cf_faith_rollout_hard_valid"),
        "cf_faith_pearl_hard_valid": _mean("cf_faith_pearl_hard_valid"),
    }


def print_summary_table(rows: list[dict]) -> None:
    hdr = (
        f"{'Method':<16}{'valid':>7}{'prox_l1':>9}{'spars':>7}{'sp_ch':>7}{'sp_tp':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
        f"{'joint_r':>9}{'joint_p':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    def _fmt(val, width):
        return f"{val:>{width}.2f}" if val is not None else " " * (width - 3) + "N/A"

    for r in rows:
        joint_r = r.get("cf_faith_rollout_hard_valid")
        joint_p = r.get("cf_faith_pearl_hard_valid")
        print(
            f"{r['method']:<16}{_fmt(r['validity'], 7)}{r['proximity_l1']:>9.3f}"
            f"{r['sparsity']:>7.2f}"
            f"{_fmt(r.get('sparsity_channels'), 7)}{_fmt(r.get('sparsity_timepoints'), 7)}"
            f"{r['cf_faith_rollout_hard']:>8.2f}{r['cf_faith_rollout_soft']:>8.2f}"
            f"{r['cf_faith_pearl_hard']:>8.2f}{r['cf_faith_pearl_soft']:>8.2f}"
            f"{_fmt(joint_r, 9)}{_fmt(joint_p, 9)}"
        )


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------------------
# Axis A (attribution/concept quality) support
# ---------------------------------------------------------------------------


def causal_parents(graph: np.ndarray, node: int) -> list[int]:
    """Ground-truth causal parents of ``node`` from the SCM's ``(k, k, L)`` graph.

    ``graph[i, j, l] == 1`` means variable *j* causes variable *i* at lag
    ``l + 1`` (see ``causaltemp_xai.benchmarks.mechanisms.MLPMechanism``
    docstring), so the parents of ``node`` are the ``j`` with any nonzero lag.
    """
    return [j for j in range(graph.shape[1]) if np.any(graph[node, j, :] != 0)]


def build_oracle_interventions(X_sel: np.ndarray, mechanism, shift: float = 1.5):
    """Ground-truth ``do(x[t0, node] = value)`` skeleton CF per instance.

    Cycles ``node = i % k`` across instances (same convention as
    ``05_run_oracle_control.py``) so every channel gets exercised. Used to
    give Axis A a *real* ``int_channel`` / causal-parent ground truth instead
    of a proxy, for any mechanism family (linear or MLP — both implement
    :func:`~causaltemp_xai.benchmarks.structural_cf.structural_counterfactual`).

    Returns
    -------
    list of ``(t0, node, x_cf_skeleton)`` tuples, one per instance in ``X_sel``.
    """
    from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual

    X_sel = np.asarray(X_sel, dtype=float)
    T, k = X_sel.shape[1], X_sel.shape[2]
    t0 = T // 2
    out = []
    for i, x in enumerate(X_sel):
        node = i % k
        value = float(x[t0, node]) + shift
        x_cf = structural_counterfactual(x, mechanism, t0, node, value, noiseless=True)
        out.append((t0, node, x_cf))
    return out


def build_masked_mechanism(mechanism, inferred_adj: np.ndarray):
    """Copy an :class:`MLPMechanism` with its adjacency replaced by ``inferred_adj``.

    Keeps the mechanism's learned weights/decay/gain but swaps in a different
    ``(k, k, L)`` parent structure. Used by the Axis-B graph-error decomposition
    (Phase 07): rolling the oracle structural CF through a mechanism that only
    propagates along the CITRIS-*inferred* edges — instead of the true edges —
    isolates how much CF-faith is lost to graph-estimation error (vs. the
    propagation error a real CF method would additionally incur).

    Raises ``TypeError`` for non-MLP mechanisms (the decomposition targets the
    nonlinear/causal benchmark, where CITRIS operates).
    """
    from causaltemp_xai.benchmarks.mechanisms import MLPMechanism

    if not isinstance(mechanism, MLPMechanism):
        raise TypeError(
            "graph-error decomposition requires an MLPMechanism (nonlinear "
            f"config); got {type(mechanism).__name__}"
        )
    inferred_adj = np.asarray(inferred_adj, dtype=float)
    if inferred_adj.shape != mechanism.graph.shape:
        raise ValueError(
            f"inferred_adj shape {inferred_adj.shape} != mechanism.graph "
            f"{mechanism.graph.shape}"
        )
    return MLPMechanism(
        graph=inferred_adj,
        hidden=mechanism.hidden,
        decay=mechanism.decay,
        gain=mechanism.gain,
        W1=mechanism.W1,
        b1=mechanism.b1,
        W2=mechanism.W2,
        b2=mechanism.b2,
        activation=mechanism.activation,
    )


def axis_a_for_attribution(attributions: np.ndarray, int_channels,
                           causal_parents_list, t0s=None) -> dict:
    """Axis A (ICC attribution-mass + causal coverage) for one attribution method.

    Thin wrapper around :func:`causaltemp_xai.metrics.axis_a.compute_axis_a`
    (``MCC_disent`` is skipped — no encoder/latent factors in this pipeline,
    only the raw saliency maps). ``t0s`` windows ICC to the post-intervention
    region (fix #6, 2026-07-18).
    """
    from causaltemp_xai.metrics.axis_a import compute_axis_a

    return compute_axis_a(
        attributions=np.asarray(attributions, dtype=float),
        int_channels=np.asarray(int_channels),
        causal_parents_list=causal_parents_list,
        t0s=None if t0s is None else np.asarray(t0s),
    )


# ---------------------------------------------------------------------------
# Axis B (graph quality) — benchmark-level diagnostic, not per-method
# ---------------------------------------------------------------------------


def axis_b_benchmark_diagnostic(graph: np.ndarray, X: np.ndarray,
                                mechanism=None) -> dict:
    """Structural diagnostic of the benchmark's own ground-truth graph.

    No graph-*discovery* method is wired into this pipeline, so there is no
    "inferred" adjacency to compare against — Axis B's SHD/LagAcc against the
    ground truth graph itself are trivially perfect (0 / 1) by construction.
    What *is* informative here is ``ResidualDep`` (metric-quality fix #1,
    2026-07-18, replacing the retired TV-confounding score): the mean |Pearson
    r| between the **mechanism residuals** of channel pairs the graph says are
    not directly connected — genuine unexplained association, ~0 for the
    benchmark's confounder-free SCMs. Pass ``mechanism`` to enable the
    residual (recommended) mode. Report once per dataset.
    """
    from causaltemp_xai.metrics.axis_b import compute_axis_b

    adj = (np.asarray(graph) != 0).astype(int)
    result = compute_axis_b(adj, adj, X=np.asarray(X, dtype=float),
                            mechanism=mechanism)
    result["note"] = (
        "SHD/LagAcc computed against the graph itself (no graph-discovery "
        "method in this pipeline) -- trivially perfect; ResidualDep is "
        "the informative benchmark-structural diagnostic here."
    )
    return result


# ---------------------------------------------------------------------------
# Multi-seed + bootstrap-CI aggregation (M2, O2)
# ---------------------------------------------------------------------------

#: Per-instance metric columns (from :func:`per_instance_records`) that get a
#: bootstrap 95% CI when pooled across seeds. Deliberately excludes bookkeeping
#: columns (``instance``, ``intervention_t``) that are not "reportable means".
SEED_AGGREGATE_METRICS: list[str] = [
    "validity",
    "proximity_l1",
    "proximity_l2",
    "sparsity",
    "sparsity_channels",
    "sparsity_timepoints",
    "trsi",
    "cf_faith_rollout_hard",
    "cf_faith_rollout_soft",
    "cf_faith_pearl_hard",
    "cf_faith_pearl_soft",
    "cf_faith_rollout_hard_valid",
    "cf_faith_pearl_hard_valid",
]


def _read_per_instance_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"no {path}; run phases 01-04 for this seed first (see "
            "experiments/06_aggregate_and_report.py seeds)"
        )
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate_across_seeds(
    base_config_name: str,
    seeds: list[int],
    classifier: str = "lstm",
    metrics: list[str] | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    boot_seed: int = 0,
    results_dir: Path = RESULTS_DIR,
) -> list[dict]:
    """Pool per-method, per-instance metrics across seed replicates and
    compute a hierarchical (seed-cluster) bootstrap 95% CI on every mean.

    Reads ``results/<base_config>_seed<seed>/<classifier>/per_instance.csv``
    for every ``seed`` in ``seeds`` (written by Phase 04, one file per seed
    replicate produced via ``causaltemp_xai.config.seeded_variant`` -- see
    ``experiments/06_aggregate_and_report.py seeds``), groups by ``method``, and for each
    metric in ``metrics`` runs :func:`causaltemp_xai.stats.hierarchical_bootstrap_ci`
    over the seed-grouped per-instance values. The hierarchical (not flat)
    bootstrap is required here specifically because each seed's instances
    share that seed's SCM draw, LSTM fit, and CF-selection order -- they are
    not i.i.d. draws across seeds (see ``causaltemp_xai/stats.py`` module
    docstring).

    Parameters
    ----------
    base_config_name:
        The registered preset name the seed replicates were derived from
        (e.g. ``"smoke"``) -- **not** a seeded name; the per-seed directories
        are reconstructed internally via ``seeded_variant``.
    seeds:
        The seed values to pool (each must have completed phases 01-04).
    classifier:
        Sub-directory under ``results/<config>/`` (``"lstm"`` for the phased
        pipeline's only classifier).
    metrics:
        Metric columns to aggregate (default :data:`SEED_AGGREGATE_METRICS`).
    n_boot, ci, boot_seed:
        Forwarded to :func:`~causaltemp_xai.stats.hierarchical_bootstrap_ci`.
        ``n_boot`` defaults lower (2000) than the stats module's own default
        (10000) so a smoke-scale multi-seed run (few methods x few metrics)
        stays fast; raise it for a final reported table.

    Returns
    -------
    list of dict
        One row per method: ``benchmark``, ``classifier``, ``method``,
        ``n_seeds``, plus ``<metric>_mean``/``<metric>_ci_lo``/
        ``<metric>_ci_hi``/``<metric>_n`` for every metric in ``metrics``.
    """
    from causaltemp_xai.config import get_config, seeded_variant
    from causaltemp_xai.stats import hierarchical_bootstrap_ci

    if metrics is None:
        metrics = SEED_AGGREGATE_METRICS

    base_cfg = get_config(base_config_name)

    per_seed_rows: dict[int, list[dict]] = {}
    for s in seeds:
        seed_cfg = seeded_variant(base_cfg, s)
        path = results_dir / seed_cfg.name / classifier / "per_instance.csv"
        per_seed_rows[s] = _read_per_instance_csv(path)

    # Union of methods across seeds, preserving first-seen order.
    methods: list[str] = []
    for s in seeds:
        for r in per_seed_rows[s]:
            if r["method"] not in methods:
                methods.append(r["method"])

    out_rows = []
    for method in methods:
        row: dict = {
            "benchmark": base_cfg.name,
            "classifier": classifier,
            "method": method,
            "n_seeds": len(seeds),
        }
        for metric in metrics:
            groups = []
            for s in seeds:
                vals = [
                    float(r[metric])
                    for r in per_seed_rows[s]
                    if r["method"] == method and r.get(metric, "") not in (None, "")
                ]
                groups.append(np.asarray(vals, dtype=float))
            result = hierarchical_bootstrap_ci(
                groups, n_boot=n_boot, ci=ci, seed=boot_seed
            )
            row.update(result.as_dict(prefix=f"{metric}_"))
        out_rows.append(row)
    return out_rows


def print_seed_aggregate_table(rows: list[dict], headline_metrics: list[str] | None = None) -> None:
    """Console summary of :func:`aggregate_across_seeds`'s output.

    Prints ``mean [ci_lo, ci_hi]`` for each of ``headline_metrics`` (default:
    validity + both CF-faith hard scores) so CI width is visible at a glance
    without opening the CSV.
    """
    if headline_metrics is None:
        headline_metrics = ["validity", "cf_faith_rollout_hard", "cf_faith_pearl_hard"]

    col_w = 16
    hdr = f"{'Method':<{col_w}}" + "".join(f"{m:>28}" for m in headline_metrics)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for m in headline_metrics:
            mean, lo, hi = r.get(f"{m}_mean"), r.get(f"{m}_ci_lo"), r.get(f"{m}_ci_hi")
            if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                cells.append(f"{'N/A':>28}")
            else:
                cells.append(f"{mean:>6.3f} [{lo:.3f},{hi:.3f}]".rjust(28))
        print(f"{r['method']:<{col_w}}" + "".join(cells))
