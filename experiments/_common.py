"""Shared helpers for the numbered experiment-phase scripts (01-08).

Every phase script writes under a single ``results/`` tree so later phases
(and ``08_aggregate_and_report.py``) can discover what earlier phases produced purely
by path convention::

    results/<config_name>/<classifier>/cf/X_sel.npy
    results/<config_name>/<classifier>/cf/X_cf_<Method>.npy
    results/<config_name>/<classifier>/eval_<Method>.json
    results/<config_name>/<classifier>/per_instance.csv
    results/<config_name>/<classifier>/shift_vr.json
    results/<config_name>/<classifier>/horizon/                (Phase 06)
    results/<config_name>/<classifier>/pns.json                (Phase 07)
    results/<config_name>/axis_a_benchmark.json      # dataset-level, no classifier needed
    results/<config_name>/oracle/...                 # nonlinear configs
    results/tables/table_axis_c_cf_faith.csv         # accumulated across runs
    results/figures/*.png

Axis routing
------------
Three axes, defined once in ``causaltemp_xai/metrics/taxonomy.py`` (that module
is the single source of truth; ``tests/test_taxonomy.py`` fails if a reported
metric has no axis). **The axes are not parallel columns** — A scores the
*benchmark*, B and C score *methods*:

* **Axis A** (SHD, lag accuracy, lagged-edge F1, graph AUC, residual
  dependence, graph-error decomposition) — per **dataset**, not per method.
  No graph-*discovery* method runs in the main pipeline, so this is computed
  once per dataset in Phase 01 as a structural diagnostic of the benchmark's
  own causal graph (:func:`axis_a_benchmark_diagnostic`). Phase 07's DYNOTEARS
  self-graphing is the one exception. Printing it beside B/C as a third
  per-method column is a category error (``docs/general_plan.md`` §5).
* **Axis B** (Shift-VR, input sensitivity) — per **method**: validity retention
  under a noise-distribution shift, computed live in Phase 03.
* **Axis C** (validity, proximity, sparsity, OOD, SCM-noise plausibility, TRSI,
  both CF-faith semantics and their gate diagnostics, the model-vs-world audit,
  do-complexity, ``frac_vacuous``, ``frac_degenerate``) — per **method**, every
  CF-*generating* method (Wachter, NoiselessSCMRecourse, cfts-*, OracleCF-*). Axis C scores the
  CF as an artifact; CF-faith scores it against the mechanism.

The oracle interventions built here (:func:`build_oracle_interventions`,
:func:`oracle_intervention_spec`) come from the known SCM, so the intervened
channel is ground truth rather than a proxy.

Superseded 2026-08-03/04: this docstring previously carried **two** conflicting
"Axis A" bullets — one describing attribution/ICC scoring, one saying Axis A is
not a per-method score. Both predated the three-axis taxonomy; the ICC/concept
axis and the attribution methods it scored were deleted, and the graph
diagnostic was relettered B -> A. ``TV-Confounding`` named here was retired with
the ``R<n>``/risk collision.
"""

from __future__ import annotations

import csv
import functools
import json
import subprocess
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


def select_flip_candidates(clf, X_test, n_cf, target_class=TARGET_CLASS, from_class=None):
    """Indices of test instances to seek counterfactuals for.

    By default (``from_class=None``) returns instances the classifier predicts
    as **not** ``target_class`` — the standard recourse direction.

    ``from_class`` selects the complementary set instead: instances predicted
    as exactly that class. Passing ``from_class=target_class`` gives the
    *reverse* direction (already in the target class, seeking a CF that leaves
    it), which is what estimating the **probability of necessity** requires —
    PN conditions on the outcome having occurred, so it cannot be estimated
    from the default flip-candidate set (see ``docs/pns_metric_design.md``).
    """
    preds = clf.predict(X_test)
    if from_class is None:
        src = [i for i in range(len(X_test)) if preds[i] != target_class]
    else:
        src = [i for i in range(len(X_test)) if preds[i] == from_class]
    if len(src) < n_cf:
        src = list(range(len(X_test)))
    return np.asarray(src[:n_cf], dtype=int)


def per_instance_records(
    benchmark,
    classifier,
    method_name,
    X_sel,
    CFs,
    graph,
    mech,
    preds=None,
    noise_scale=None,
    no_cf_found=None,
):
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
    None. The separately named ``*_hard_given_valid`` columns are populated
    only for valid instances so their aggregate is ``P(hard | valid)``.

    ``vacuous`` (RISK-17, 2026-07-31) is 1 when the CF encodes no intervention
    at all: ``x_cf`` differs from ``x``, but ``x_cf[intervention_t]`` is exactly
    what the mechanism predicts from ``x_cf``'s own prefix. A noiseless-rollout
    CF with a zero perturbation is the case in point — it drops the factual
    noise from ``t0`` onward, so it looks like a large edit and scores
    ``cf_faith_rollout_hard = 1.0`` while having done nothing. Read it before
    the CF-faith and proximity columns on the same row, not after.
    """
    from causaltemp_xai.metrics.axis_c import (
        proximity,
        scm_noise_plausibility,
        sparsity,
        trsi,
    )
    from causaltemp_xai.metrics.cf_faith import CFfaith
    from causaltemp_xai.metrics.pns import do_complexity
    from causaltemp_xai.scm.intervention import derive_intervention_t, is_vacuous_intervention

    def _nan_to_zero(v):
        return 0.0 if np.isnan(v) else v

    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    if no_cf_found is None:
        no_cf_found_a = np.zeros(len(CFs), dtype=bool)
    else:
        no_cf_found_a = np.asarray(no_cf_found)
        if no_cf_found_a.dtype != np.bool_:
            raise TypeError(f"no_cf_found must have bool dtype, got {no_cf_found_a.dtype}")
        if no_cf_found_a.shape != (len(CFs),):
            raise ValueError(
                f"no_cf_found shape {no_cf_found_a.shape} does not match CF batch ({len(CFs)},)"
            )
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mech)
        p = pearl.score(x, x_cf, t, graph, mech)
        _spars_detail = sparsity(x, x_cf, return_detailed=True)
        valid_i = int(preds[i] == TARGET_CLASS) if preds is not None else None
        do_c = do_complexity(x, x_cf, mech)
        rows.append(
            {
                "benchmark": benchmark,
                "classifier": classifier,
                "method": method_name,
                "instance": int(i),
                "validity": (valid_i if valid_i is not None else ""),
                # Generator search outcome. Kept separate from classifier
                # validity and mechanism-level vacuity by design.
                "no_cf_found": int(no_cf_found_a[i]),
                "proximity_l1": proximity(x, x_cf, norm="l1"),
                "proximity_l2": proximity(x, x_cf, norm="l2"),
                "sparsity": sparsity(x, x_cf),
                "sparsity_channels": _spars_detail["channels"],
                "sparsity_timepoints": _spars_detail["timepoints"],
                "trsi": trsi(x_cf, x),
                # Ground-truth plausibility: abduct the noise the CF *implies*
                # under the true mechanism and compare its scale to the SCM's.
                # Unlike `ood` (IsolationForest, a mechanism-free stand-in) this
                # needs no estimator, and its symmetric log-ratio penalises the
                # noiseless skeleton as well as over-large edits. Blank when the
                # caller does not supply the config's noise scale.
                "scm_noise_plausibility": (
                    scm_noise_plausibility(x_cf, mech, noise_scale) if noise_scale else ""
                ),
                "intervention_t": int(t),
                # RISK-17: 1 when the CF encodes no do() at all -- x_cf[t] is
                # what the mechanism predicts from x_cf's own prefix, so the
                # departure from the factual is continuation, not action. A
                # zero-perturbation noiseless rollout scores rollout_hard=1.0
                # while being vacuous; frac_degenerate does not catch it.
                "vacuous": int(is_vacuous_intervention(x, x_cf, mech, t0=t)),
                # Pearl-semantic schedule length. This is a distinct predicate
                # from the noiseless-semantic vacuity flag above.
                "do_complexity": int(do_c),
                "cf_faith_rollout_hard": r["hard"],
                "cf_faith_rollout_soft": r["soft"],
                "cf_faith_pearl_hard": p["hard"],
                "cf_faith_pearl_soft": p["soft"],
                # Joint faithfulness-validity (anti-gameability, M1 2026-07-07).
                # A degenerate (NaN) hard score counts as 0 here, not NaN:
                # this is a fraction-of-batch criterion, so an instance whose
                # faithfulness cannot be established must not be credited.
                "cf_faith_rollout_hard_valid": (
                    _nan_to_zero(r["hard"]) * valid_i if valid_i is not None else ""
                ),
                "cf_faith_pearl_hard_valid": (
                    _nan_to_zero(p["hard"]) * valid_i if valid_i is not None else ""
                ),
                # Invalid rows abstain from this conditional denominator;
                # degenerate valid rows count as zero conditional credit.
                "cf_faith_rollout_hard_given_valid": (
                    _nan_to_zero(r["hard"]) if valid_i == 1 else ""
                ),
                "cf_faith_pearl_hard_given_valid": (
                    _nan_to_zero(p["hard"]) if valid_i == 1 else ""
                ),
            }
        )
    return rows


def score_and_collect(
    cfg,
    slot,
    method_name,
    X_sel,
    cfs,
    graph,
    mech,
    clf=None,
    extra=None,
    no_cf_found=None,
):
    """Score one method's counterfactuals: per-instance rows + the aggregate row.

    The step phases 04, 05 and 06 all perform identically, differing only in
    three things this signature makes explicit:

    * ``clf=None`` — callers may omit classifier-dependent outcome metrics.
      Phase 05 now supplies the trained classifier so the structural oracle
      remains a construction-level positive control while also calibrating the
      same outcome-quality columns as every other method.
    * ``slot`` — the ``results/<config>/<slot>/`` directory: ``"lstm"`` for the
      explainer phases, ``"oracle"`` for the control.
    * ``extra`` — extra keys stamped onto *both* the per-instance rows and the
      aggregate, which Phase 06 uses for ``horizon`` / ``t0`` / ``t_label``.

    ``noise_scale`` is derived here from ``cfg.noise_type`` rather than passed
    in: all three callers computed the identical
    ``expected_abs_noise(cfg.noise_type)``, and the families are variance-matched
    so the value genuinely differs per family (Laplace 0.100, Uniform 0.085,
    Gaussian 0.113) — not something a caller should be able to get wrong.

    Deliberately does **not** write anything. The three phases differ in where
    their output goes and in whether they append to the committed cross-run
    table (Phase 06 must not — see its docstring), so unifying the writes would
    erase a distinction that is load-bearing.
    """
    from causaltemp_xai.benchmarks.generator import expected_abs_noise

    preds = None if clf is None else np.asarray(clf.predict(cfs)).reshape(-1)
    rows = per_instance_records(
        cfg.name,
        slot,
        method_name,
        X_sel,
        cfs,
        graph,
        mech,
        preds,
        noise_scale=expected_abs_noise(cfg.noise_type),
        no_cf_found=no_cf_found,
    )
    agg = aggregate_method_row(cfg.name, slot, method_name, rows)
    if extra:
        for r in rows:
            r.update(extra)
        agg.update(extra)
    return rows, agg


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
        # NaN-aware: CF-faith is NaN on degenerate instances (intervention_t
        # >= T-1, see CFfaith) and those must abstain from the mean rather
        # than poison it to NaN or count as a free 1.0.
        vals = [
            float(r[key])
            for r in instance_rows
            if r.get(key) not in (None, "") and not np.isnan(float(r[key]))
        ]
        return float(np.mean(vals)) if vals else None

    n = len(instance_rows)
    n_scorable = sum(
        1
        for r in instance_rows
        if r.get("cf_faith_rollout_hard") not in (None, "")
        and not np.isnan(float(r["cf_faith_rollout_hard"]))
    )
    # RISK-17. Tolerates rows predating the column (blank -> not counted)
    # rather than raising, so an older per_instance.csv still aggregates.
    n_vacuous = sum(
        1 for r in instance_rows if str(r.get("vacuous", "")).strip() not in ("", "0", "False")
    )
    n_no_cf_found = sum(
        1 for r in instance_rows if str(r.get("no_cf_found", "")).strip() not in ("", "0", "False")
    )
    do_values = [
        float(r["do_complexity"])
        for r in instance_rows
        if r.get("do_complexity") not in (None, "") and not np.isnan(float(r["do_complexity"]))
    ]
    do_scorable_values = [value for value in do_values if value > 0]
    n_do_scorable = len(do_scorable_values)
    do_mean_all = float(np.mean(do_values)) if do_values else float("nan")
    do_mean_scorable = float(np.mean(do_scorable_values)) if do_scorable_values else float("nan")
    has_validity = any(r.get("validity") not in (None, "") for r in instance_rows)

    def _conditional_mean(key):
        value = _mean(key)
        if value is not None:
            return value
        return float("nan") if has_validity else None

    return {
        "benchmark": benchmark,
        "classifier": classifier,
        "method": method_name,
        "n": n,
        "validity": _mean("validity"),
        "proximity_l1": _mean("proximity_l1"),
        "proximity_l2": _mean("proximity_l2"),
        "sparsity": _mean("sparsity"),
        "sparsity_channels": _mean("sparsity_channels"),
        "sparsity_timepoints": _mean("sparsity_timepoints"),
        "trsi": _mean("trsi"),
        # Ground-truth plausibility (None when the caller supplied no noise
        # scale, e.g. rows aggregated from a pre-2026-08-03 per_instance.csv).
        "scm_noise_plausibility": _mean("scm_noise_plausibility"),
        "cf_faith_rollout_hard": _mean("cf_faith_rollout_hard"),
        "cf_faith_rollout_soft": _mean("cf_faith_rollout_soft"),
        "cf_faith_pearl_hard": _mean("cf_faith_pearl_hard"),
        "cf_faith_pearl_soft": _mean("cf_faith_pearl_soft"),
        # Degeneracy diagnostics — a high frac_degenerate invalidates the
        # CF-faith columns on this row regardless of their value.
        "n_cf_faith_scorable": n_scorable,
        "frac_degenerate": (float(1.0 - n_scorable / n) if n else None),
        # Vacuity diagnostic (RISK-17) — a high frac_vacuous invalidates this
        # row's cf_faith_rollout_* and proximity/sparsity columns regardless
        # of their value: the CFs contain no intervention to score.
        "n_vacuous": n_vacuous,
        "frac_vacuous": (float(n_vacuous / n) if n else None),
        "n_no_cf_found": n_no_cf_found,
        "frac_no_cf_found": (float(n_no_cf_found / n) if n else None),
        "do_complexity_mean_all": do_mean_all,
        "do_complexity_mean_pearl_scorable": do_mean_scorable,
        "n_do_scorable": n_do_scorable,
        "frac_no_do_schedule": (float(1.0 - n_do_scorable / n) if n else None),
        # Migration alias: always the all-instance Pearl-semantic mean.
        "do_complexity_mean": do_mean_all,
        # Joint faithfulness-validity (None for classifier-free oracle rows).
        "cf_faith_rollout_hard_valid": _mean("cf_faith_rollout_hard_valid"),
        "cf_faith_pearl_hard_valid": _mean("cf_faith_pearl_hard_valid"),
        "cf_faith_rollout_hard_given_valid": _conditional_mean("cf_faith_rollout_hard_given_valid"),
        "cf_faith_pearl_hard_given_valid": _conditional_mean("cf_faith_pearl_hard_given_valid"),
    }


def print_summary_table(rows: list[dict]) -> None:
    hdr = (
        f"{'Method':<16}{'valid':>7}{'prox_l1':>9}{'spars':>7}{'sp_ch':>7}{'sp_tp':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
        f"{'joint_r':>9}{'joint_p':>9}{'vac':>7}"
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
            f"{_fmt(joint_r, 9)}{_fmt(joint_p, 9)}{_fmt(r.get('frac_vacuous'), 7)}"
        )


# ---------------------------------------------------------------------------
# Run provenance (R7) — seed + commit hash stamped into every result file
# ---------------------------------------------------------------------------
#
# R7 requires a ``seed`` field in every result file, and the M2 DoD additionally
# requires the commit hash. Both were previously written by hand into the
# ``provenance`` block of ``summary.json`` only, so 114 of 129 result JSONs
# carried neither -- a per-call-site convention is something a new phase forgets
# by default. Stamping inside :func:`dump_json` instead makes the gate hold for
# every current *and* future phase without anyone remembering to opt in.
#
# Deliberately NOT stamped: a wall-clock timestamp. Result JSONs are tracked in
# git, so a timestamp would make every re-run produce a diff even when the
# numbers are identical, and "this file changed" would stop meaning "these
# results changed".

_RUN_CONTEXT: dict = {"seed": None, "config": None}


def set_run_context(seed=None, config: str | None = None) -> None:
    """Declare the seed/config the current phase is running under.

    Every subsequent :func:`dump_json` call in this process stamps them into the
    file it writes. Phases call this once, right after resolving their config.
    """
    if seed is not None:
        _RUN_CONTEXT["seed"] = int(seed)
    if config is not None:
        _RUN_CONTEXT["config"] = str(config)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def git_provenance() -> dict:
    """``{"git_commit": <sha|None>, "git_dirty": <bool|None>}`` for this checkout.

    ``git_dirty`` is load-bearing for reproducibility honesty: a commit hash
    recorded while the working tree had uncommitted changes does not identify
    the code that produced the numbers, so the flag says so rather than letting
    the hash imply a cleanliness it does not have. Cached -- the answer cannot
    change within a single phase run.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"git_commit": None, "git_dirty": None}
    status = _git("status", "--porcelain")
    return {"git_commit": sha, "git_dirty": bool(status)}


def run_provenance() -> dict:
    """Seed + config + commit provenance stamped into every result file."""
    return {
        "seed": _RUN_CONTEXT["seed"],
        "config": _RUN_CONTEXT["config"],
        **git_provenance(),
    }


def dump_json(path: Path, obj) -> None:
    """Write ``obj`` as JSON, stamping run provenance (R7) into dict payloads.

    Existing keys are never overwritten -- a phase that already wrote its own
    ``seed``/``config`` keeps them, so this only ever fills gaps.
    """
    if isinstance(obj, dict):
        obj = {**{k: v for k, v in run_provenance().items() if k not in obj}, **obj}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------------------
# Provenance auditing for downstream consumers (08_aggregate_and_report.py)
# ---------------------------------------------------------------------------
#
# ``dump_json`` stamps ``git_dirty`` honestly into every result JSON, but the
# two report paths in Phase 08 (``figures``, ``seeds``) read only
# ``per_instance.csv`` -- a plain CSV with no provenance columns -- so a
# dirty-run warning written into a sibling ``summary.json`` was reachable
# per-file but silently unreachable from the one place that turns results into
# publication artifacts. This section makes that flag visible again at the
# point where it matters (added 2026-07-31; see ``docs/risk_register.md``
# RISK-13).


def read_run_summary_provenance(summary_path: Path) -> dict | None:
    """Read ``seed``/``git_commit``/``git_dirty`` out of a ``summary.json``.

    Returns ``None`` if the file is missing or unreadable. A ``per_instance.csv``
    with no sibling ``summary.json`` (or one predating R7 provenance stamping)
    has no traceable provenance at all -- reported as ``"missing"`` by
    :func:`check_provenance` rather than silently skipped.
    """
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    return {
        "seed": obj.get("seed"),
        "git_commit": obj.get("git_commit"),
        "git_dirty": obj.get("git_dirty"),
    }


def check_provenance(result_dirs: list[Path]) -> list[dict]:
    """Provenance report for a list of ``results/<config>/<classifier>`` dirs.

    Each directory is expected to hold a Phase-04-style ``summary.json``
    (stamped via :func:`dump_json`, so it carries ``git_dirty`` whenever a
    commit hash was resolvable at all). One row per directory::

        {"dir": str, "status": "clean" | "dirty" | "missing",
         "seed": ..., "git_commit": ..., "git_dirty": ...}

    ``"dirty"`` means the result was produced while the working tree had
    uncommitted changes, so ``git_commit`` does not identify the code that
    produced it (see :func:`git_provenance`). ``"missing"`` means no
    ``summary.json`` / no commit hash was found at all -- e.g. git was
    unavailable, or the file predates R7 stamping.
    """
    report = []
    for d in result_dirs:
        prov = read_run_summary_provenance(Path(d) / "summary.json")
        if prov is None or prov.get("git_commit") is None:
            status = "missing"
        elif prov.get("git_dirty"):
            status = "dirty"
        else:
            status = "clean"
        report.append({"dir": str(d), "status": status, **(prov or {})})
    return report


def print_provenance_warning(report: list[dict]) -> None:
    """Print a loud banner if any directory in ``report`` is not clean.

    A figure or aggregated table built from a dirty or unprovenanced run is
    byte-for-byte indistinguishable from one built from a clean, committed
    run unless someone opens every ``summary.json`` by hand -- so silence here
    is exactly the failure mode this function exists to prevent. Does not
    raise: the artifacts are still written (useful for fast local iteration),
    but the warning must be impossible to miss in the console output.
    """
    bad = [r for r in report if r["status"] != "clean"]
    if not bad:
        return
    print("\n" + "=" * 78)
    print("[08] PROVENANCE WARNING -- not every input is from a clean, committed state")
    print("=" * 78)
    for r in bad:
        if r["status"] == "missing":
            print(f"  MISSING  {r['dir']}  (no summary.json / no commit hash found)")
        else:
            print(
                f"  DIRTY    {r['dir']}  git_commit={r.get('git_commit')}  "
                "-- working tree had uncommitted changes when this ran; the "
                "commit hash does not identify the code that produced it"
            )
    print("=" * 78)
    print(
        "[08] Figures/tables are still written from this data, but do not cite "
        "them in a manuscript until every input reads 'clean' -- re-run the "
        "affected phase(s) once the working tree is committed. See "
        "docs/risk_register.md RISK-13."
    )
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Oracle-intervention support (ground-truth do() specs from the known SCM)
# ---------------------------------------------------------------------------


def causal_parents(graph: np.ndarray, node: int) -> list[int]:
    """Ground-truth causal parents of ``node`` from the SCM's ``(k, k, L)`` graph.

    ``graph[i, j, l] == 1`` means variable *j* causes variable *i* at lag
    ``l + 1`` (see ``causaltemp_xai.benchmarks.mechanisms.MLPMechanism``
    docstring), so the parents of ``node`` are the ``j`` with any nonzero lag.
    """
    return [j for j in range(graph.shape[1]) if np.any(graph[node, j, :] != 0)]


#: Magnitude of the ground-truth ``do()`` used by the oracle control and by the
#: Axis-A intervention fixtures, in units of the channel's own value.
#:
#: **One definition, deliberately.** This lived in three places until 2026-08-04
#: — here as a default, and as a module constant in both
#: ``05_run_oracle_control.py`` and ``07_auxiliary_methods.py``, the latter
#: carrying the comment *"must match ``build_oracle_interventions``' default"*.
#: Duplication that has to document its own fragility is a defect: changing this
#: number would silently have desynchronised the oracle control from the fixtures
#: it is supposed to anchor.
ORACLE_SHIFT = 1.5


def oracle_intervention_spec(X_sel: np.ndarray, k: int, shift: float = ORACLE_SHIFT):
    """The benchmark's ground-truth ``do()`` convention, in one place.

    Yields ``(t0, node, value)`` per instance: intervene at the midpoint
    ``t0 = T // 2``, cycling ``node = i % k`` so every channel gets exercised,
    setting it to its own value plus ``shift``.

    Callers differ in what they *build* from this — the oracle control wants an
    ``(N, T, k)`` batch and parameterises ``noiseless``, while the Axis-A
    fixtures want ``(t0, node, skeleton)`` tuples — but the convention itself
    must not differ, or the control stops anchoring the thing it anchors.
    """
    X_sel = np.asarray(X_sel, dtype=float)
    t0 = X_sel.shape[1] // 2
    for i, x in enumerate(X_sel):
        node = i % k
        yield t0, node, float(x[t0, node]) + shift


def build_oracle_interventions(X_sel: np.ndarray, mechanism, shift: float = ORACLE_SHIFT):
    """Ground-truth ``do(x[t0, node] = value)`` skeleton CF per instance.

    Uses :func:`oracle_intervention_spec` for the convention. Gives Axis A a
    *real* ``int_channel`` / causal-parent ground truth instead of a proxy, for
    any mechanism family (linear or MLP — both implement
    :func:`~causaltemp_xai.benchmarks.structural_cf.structural_counterfactual`).

    Returns
    -------
    list of ``(t0, node, x_cf_skeleton)`` tuples, one per instance in ``X_sel``.
    """
    from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual

    X_sel = np.asarray(X_sel, dtype=float)
    return [
        (t0, node, structural_counterfactual(x, mechanism, t0, node, value, noiseless=True))
        for x, (t0, node, value) in zip(
            X_sel, oracle_intervention_spec(X_sel, X_sel.shape[2], shift)
        )
    ]


def build_masked_mechanism(mechanism, inferred_adj: np.ndarray):
    """Copy an :class:`MLPMechanism` with its adjacency replaced by ``inferred_adj``.

    Keeps the mechanism's learned weights/decay/gain but swaps in a different
    ``(k, k, L)`` parent structure. Used by the Axis-A graph-error decomposition
    (Phase 07): rolling the oracle structural CF through a mechanism that only
    propagates along the method's *inferred* edges — instead of the true edges —
    isolates how much CF-faith is lost to graph-estimation error (vs. the
    propagation error a real CF method would additionally incur).

    **Extended to the M4c non-dissipative families 2026-08-05.** The
    graph-quality ladder measured on ``full_nl`` is nearly flat (a fully random
    graph costs 0.0033 of CF-faith, ~54x less range than the method axis), and
    the standing explanation is that a *dissipative* mechanism attenuates the
    oracle shift before parent-set differences can propagate. Testing that
    requires running the same ladder on a family where effects persist
    (``rho ~ 1``), so ``SpringMechanism`` is supported here. Masking is arguably
    more direct for it than for the MLP: it takes ``graph`` as an explicit
    coupling structure, so restricting it zeroes the corresponding couplings and
    nothing else -- the springs' stiffness ``k_spring``/``dt`` is carried through
    unchanged.

    ``KuramotoMechanism`` was supported here until 2026-08-11, when the Kuramoto
    family was removed from the project.

    Raises ``TypeError`` for any other mechanism family.
    """
    from causaltemp_xai.benchmarks.mechanisms import MLPMechanism, SpringMechanism

    if not isinstance(mechanism, (MLPMechanism, SpringMechanism)):
        raise TypeError(
            "graph-error decomposition requires an MLPMechanism or "
            f"SpringMechanism; got {type(mechanism).__name__}"
        )
    inferred_adj = np.asarray(inferred_adj, dtype=float)
    if inferred_adj.shape != mechanism.graph.shape:
        raise ValueError(
            f"inferred_adj shape {inferred_adj.shape} != mechanism.graph "
            f"{mechanism.graph.shape}"
        )
    if isinstance(mechanism, SpringMechanism):
        return SpringMechanism(graph=inferred_adj, k_spring=mechanism.k_spring, dt=mechanism.dt)
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


def axis_a_benchmark_diagnostic(graph: np.ndarray, X: np.ndarray, mechanism=None) -> dict:
    """Structural diagnostic of the benchmark's own ground-truth graph.

    No graph-*discovery* method is wired into this pipeline, so there is no
    "inferred" adjacency to compare against — Axis A's SHD/LagAcc against the
    ground truth graph itself are trivially perfect (0 / 1) by construction.
    What *is* informative here is ``ResidualDep`` (metric-quality fix #1,
    2026-07-18, replacing the retired TV-confounding score): the mean |Pearson
    r| between the **mechanism residuals** of channel pairs the graph says are
    not directly connected — genuine unexplained association, ~0 for the
    benchmark's confounder-free SCMs. Pass ``mechanism`` to enable the
    residual (recommended) mode. Report once per dataset.
    """
    from causaltemp_xai.metrics.axis_a import compute_axis_a

    adj = (np.asarray(graph) != 0).astype(int)
    result = compute_axis_a(adj, adj, X=np.asarray(X, dtype=float), mechanism=mechanism)
    result["note"] = (
        "SHD/LagAcc computed against the graph itself (no graph-discovery "
        "method in this pipeline) -- trivially perfect; ResidualDep is "
        "the informative benchmark-structural diagnostic here."
    )
    return result


# ---------------------------------------------------------------------------
# Canonical CF-method roster
# ---------------------------------------------------------------------------

#: The canonical registry keys written by ``03_run_cf_methods.build_methods()``.
#:
#: Phase 04 discovers methods by globbing ``cf/X_cf_*.npy`` and taking the
#: filename stem, so **any** stale array on disk is silently re-scored as a live
#: method on every re-run. That is not hypothetical: the M3 rename of
#: ``CausalFeasibility`` -> ``TSCausal`` (2026-08-04) left the pre-rename arrays
#: in place, and phase 04 kept emitting a duplicate ``eval_CausalFeasibility.json``
#: plus a full set of duplicate ``per_instance.csv`` rows -- the same method
#: double-counted under two names in every affected table.
#:
#: ``tests/test_experiment_registry.py`` asserts this set equals
#: ``build_methods()``'s keys, so adding a method to the registry without
#: updating this constant fails the suite rather than silently dropping the new
#: method from evaluation.
CF_METHOD_KEYS: frozenset[str] = frozenset(
    {
        "NoiselessSCMRecourse",
        "PearlSCMRecourse",
        "CftsWachter",
        "CftsCOMTE",
        "CftsConfeti",
        "CftsCounts",
        "CftsCels",
        "TSCausal",
    }
)


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
    "scm_noise_plausibility",
    "cf_faith_rollout_hard",
    "cf_faith_rollout_soft",
    "cf_faith_pearl_hard",
    "cf_faith_pearl_soft",
    "cf_faith_rollout_hard_valid",
    "cf_faith_pearl_hard_valid",
    "cf_faith_rollout_hard_given_valid",
    "cf_faith_pearl_hard_given_valid",
    "do_complexity",
]


def _read_per_instance_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"no {path}; run phases 01-04 for this seed first (see "
            "experiments/08_aggregate_and_report.py seeds)"
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
    ``experiments/08_aggregate_and_report.py seeds``), groups by ``method``, and for each
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
            result = hierarchical_bootstrap_ci(groups, n_boot=n_boot, ci=ci, seed=boot_seed)
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
