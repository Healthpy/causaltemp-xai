"""Phase 04: Score persisted CF arrays with Axis-C + both CF-faith semantics.

Reads ``results/<config>/lstm/cf/X_sel.npy`` and every
``results/<config>/lstm/cf/X_cf_<Method>.npy`` written by Phase 03, scores
each method with :func:`causaltemp_xai.eval.evaluate_method` plus the
per-instance Axis-C extras (TRSI -- see ``experiments/_common.py``), and
writes:

    results/<config>/lstm/eval_<Method>.json    per-method batch-mean metrics (Axis C + CF-faith)
    results/<config>/lstm/per_instance.csv      one row per (method, instance)
    results/<config>/lstm/summary.json          combined provenance + methods + axis_a + shift_vr
    results/tables/table_axis_c_cf_faith.csv    appended, one row per method (cross-run table)

CF-*generating* methods (the ones this phase scores) are evaluated on **Axis C**
(validity, proximity, sparsity, OOD, SCM-noise plausibility, TRSI, both CF-faith
semantics, the model-vs-world audit and do-complexity -- see
``causaltemp_xai/metrics/taxonomy.py``). This phase computes none of the other
axes itself; it folds their already-written JSON into ``summary.json`` for one
combined view: **Axis A** (the per-*dataset* graph diagnostic) from Phase 01,
and **Axis B**'s Shift-VR from Phase 03.

Only artifacts a phase still writes are folded in. An ``attribution`` key was
read here until 2026-08-04 from a file nothing had produced since ``ba8e990``
(RISK-21).

Usage
-----
    uv run python experiments/04_evaluate_axes.py --config smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.eval import evaluate_method  # noqa: E402
from causaltemp_xai.metrics.taxonomy import AXIS_METRICS  # noqa: E402
from experiments._common import (  # noqa: E402
    CF_METHOD_KEYS,
    RESULTS_DIR,
    TABLES_DIR,
    append_table,
    config_dir,
    dump_json,
    print_summary_table,
    score_and_collect,
    set_run_context,
    write_csv,
)

STATUS_REQUIRED_METHODS = frozenset({"NoiselessSCMRecourse", "PearlSCMRecourse"})


def load_no_cf_found(cf_dir: Path, method_name: str, n: int) -> np.ndarray:
    """Load and strictly validate a Phase-03 failed-search sidecar.

    The two SCM recourse controls have a native status API, so their sidecars are mandatory.
    Older artifacts for other methods fall back to all-false inferred status.
    """
    path = cf_dir / f"no_cf_found_{method_name}.npy"
    if not path.exists():
        if method_name in STATUS_REQUIRED_METHODS:
            raise SystemExit(
                f"missing {path}; rerun Phase 03 so {method_name} search failures "
                "cannot be silently scored as successful searches"
            )
        print(
            f"[04] {method_name}: no failed-search sidecar; inferring all false for old artifact",
            file=sys.stderr,
        )
        return np.zeros(n, dtype=bool)

    status = np.load(path, allow_pickle=False)
    if status.dtype != np.bool_:
        raise SystemExit(f"{path} must have bool dtype, got {status.dtype}")
    if status.shape != (n,):
        raise SystemExit(f"{path} has shape {status.shape}; expected ({n},)")
    return status


def run(config_name: str, out_dir, seed: int | None = None) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    set_run_context(seed=cfg.seed, config=cfg.name)
    data = load_dataset(cfg.name, out_dir=out_dir)
    graph, mech = data["graph"], data["mechanism"]

    res_dir = config_dir(cfg.name, "lstm")
    cf_dir = res_dir / "cf"
    x_sel_path = cf_dir / "X_sel.npy"
    if not x_sel_path.exists():
        raise SystemExit(
            f"no {x_sel_path}; run first: uv run python experiments/03_run_cf_methods.py "
            f"--config {cfg.name}"
        )
    X_sel = np.load(x_sel_path)

    cf_paths = sorted(cf_dir.glob("X_cf_*.npy"))
    if not cf_paths:
        raise SystemExit(f"no X_cf_*.npy found under {cf_dir}")

    from causaltemp_xai.classifiers import LSTMClassifier

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    clf = LSTMClassifier.load(ckpt)

    status_provenance_path = cf_dir / "no_cf_found_provenance.json"
    if status_provenance_path.exists():
        with open(status_provenance_path) as fh:
            status_provenance = json.load(fh).get("methods", {})
    else:
        status_provenance = {}

    summary, all_instance_rows, table_rows = [], [], []
    status_sources: dict[str, str] = {}
    for cf_path in cf_paths:
        method_name = cf_path.stem[len("X_cf_") :]
        # Method discovery is filename-driven, so a CF array left behind by a
        # renamed or removed method would otherwise be re-scored forever as if
        # it were live -- double-counting one method under two names. Skip
        # loudly rather than silently: a stale array must not block the run, but
        # it must never reach per_instance.csv or the cross-run tables.
        if method_name not in CF_METHOD_KEYS:
            print(
                f"[04] SKIP {method_name!r}: not in the CF method registry "
                f"({sorted(CF_METHOD_KEYS)}). Stale array from a renamed or "
                f"removed method -- delete {cf_path.name} to silence this.",
                file=sys.stderr,
            )
            continue
        cfs = np.load(cf_path)
        no_cf_found = load_no_cf_found(cf_dir, method_name, len(cfs))
        status_sources[method_name] = status_provenance.get(method_name, {}).get(
            "source",
            "sidecar" if (cf_dir / f"no_cf_found_{method_name}.npy").exists() else "inferred",
        )
        rec = evaluate_method(
            clf,
            X_sel,
            cfs,
            data["X_train"],
            graph,
            mech,
            target_class=1,
            no_cf_found=no_cf_found,
        )
        rec["method"] = method_name

        instance_rows, agg_row = score_and_collect(
            cfg,
            "lstm",
            method_name,
            X_sel,
            cfs,
            graph,
            mech,
            clf=clf,
            no_cf_found=no_cf_found,
        )
        all_instance_rows.extend(instance_rows)
        table_rows.append(agg_row)
        # Fill in every Axis-C metric the aggregator computes but
        # ``evaluate_method`` does not (currently ``trsi`` and
        # ``scm_noise_plausibility``). Driven by the taxonomy rather than
        # patched key-by-key: a one-off ``rec["trsi"] = ...`` line is exactly
        # how ``scm_noise_plausibility`` came to be computed per instance yet
        # absent from every summary.json.
        for _k in AXIS_METRICS["C"]:
            if _k not in rec and _k in agg_row:
                rec[_k] = agg_row[_k]
        summary.append(rec)

        eval_path = res_dir / f"eval_{method_name}.json"
        dump_json(eval_path, rec)
        print(
            f"[04] {method_name:<14} validity={rec['validity']:.2f} "
            f"prox_l1={rec['proximity_l1']:.3f} "
            f"cf_faith_rollout_hard={rec['cf_faith_rollout_hard']:.2f} "
            f"cf_faith_pearl_hard={rec['cf_faith_pearl_hard']:.2f} "
            f"frac_vacuous={rec['frac_vacuous']:.2f} "
            f"frac_no_cf_found={rec['frac_no_cf_found']:.2f}"
        )

    write_csv(res_dir / "per_instance.csv", all_instance_rows)
    append_table(TABLES_DIR / "table_axis_c_cf_faith.csv", table_rows)

    def _load_if_exists(path: Path):
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
        return None

    shift = _load_if_exists(res_dir / "shift_vr.json")
    axis_a = _load_if_exists(RESULTS_DIR / cfg.name / "axis_a_benchmark.json")

    # NB: only read artifacts some phase still *writes*. An `attribution` key was
    # loaded here until 2026-08-04 from `attribution.json`, which no phase has
    # produced since the attribution block was removed (`ba8e990`) — so it went on
    # silently merging pre-removal files into freshly regenerated summaries, which
    # a clean checkout could not reproduce (RISK-21).
    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "seed": cfg.seed,
            "n_cf": len(X_sel),
            "no_cf_found_status_source": status_sources,
        },
        "methods": summary,  # Axis C + CF-faith, per CF method
        "axis_a": axis_a,  # Axis A: dataset graph diagnostic (Phase 01)
        "shift_vr": shift,  # Axis B: CF-method validity retention (Phase 03)
    }
    dump_json(res_dir / "summary.json", results)

    print()
    print_summary_table(summary)
    print(f"\n[04] wrote {res_dir / 'summary.json'}")
    print(f"[04] wrote {res_dir / 'per_instance.csv'}")
    print(f"[04] appended {TABLES_DIR / 'table_axis_c_cf_faith.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Axis-C + CF-faith on persisted CF arrays."
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Multi-seed replicate (M2): override --config's registered seed "
            "via causaltemp_xai.config.seeded_variant, reading/writing under "
            "'<config>_seed<seed>'."
        ),
    )
    args = parser.parse_args(argv)
    run(args.config, args.out_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
