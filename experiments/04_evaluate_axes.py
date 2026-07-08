"""Phase 04: Score persisted CF arrays with Axis-C + both CF-faith semantics.

Reads ``results/<config>/lstm/cf/X_sel.npy`` and every
``results/<config>/lstm/cf/X_cf_<Method>.npy`` written by Phase 03, scores
each method with :func:`causaltemp_xai.eval.evaluate_method` plus the
per-instance Axis-C extras (TRSI, IVR -- see ``experiments/_common.py``), and
writes:

    results/<config>/lstm/eval_<Method>.json    per-method batch-mean metrics (Axis C + CF-faith)
    results/<config>/lstm/per_instance.csv      one row per (method, instance)
    results/<config>/lstm/summary.json          combined provenance + methods + axis_a + axis_b + attribution + shift_vr
    results/tables/table_axis_c_cf_faith.csv    appended, one row per method (cross-run table)

CF-*generating* methods (the ones this phase scores) are evaluated on **Axis
C** (validity/proximity/sparsity/OOD/TRSI/IVR) and **CF-faith**. Axis A
(attribution quality) and Axis D's Shift-VR/input-sensitivity are computed in
Phase 03 for the attribution method / native CF methods respectively -- this
phase folds their already-written JSON into ``summary.json`` for one combined
view. Axis B is a per-*dataset* diagnostic written by Phase 01.

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
from experiments._common import (  # noqa: E402
    RESULTS_DIR,
    TABLES_DIR,
    aggregate_method_row,
    append_table,
    config_dir,
    dump_json,
    per_instance_records,
    print_summary_table,
    write_csv,
)


def run(config_name: str, out_dir, seed: int | None = None) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
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

    summary, all_instance_rows, table_rows = [], [], []
    for cf_path in cf_paths:
        method_name = cf_path.stem[len("X_cf_"):]
        cfs = np.load(cf_path)
        rec = evaluate_method(clf, X_sel, cfs, data["X_train"], graph, mech, target_class=1)
        rec["method"] = method_name

        preds = np.asarray(clf.predict(cfs)).reshape(-1)
        instance_rows = per_instance_records(cfg.name, "lstm", method_name, X_sel, cfs, graph, mech, preds)
        all_instance_rows.extend(instance_rows)
        agg_row = aggregate_method_row(cfg.name, "lstm", method_name, instance_rows)
        table_rows.append(agg_row)
        rec["trsi"] = agg_row["trsi"]  # Axis C extras not covered by evaluate_method
        rec["ivr"] = agg_row["ivr"]
        summary.append(rec)

        eval_path = res_dir / f"eval_{method_name}.json"
        dump_json(eval_path, rec)
        print(
            f"[04] {method_name:<14} validity={rec['validity']:.2f} "
            f"prox_l1={rec['proximity_l1']:.3f} "
            f"cf_faith_rollout_hard={rec['cf_faith_rollout_hard']:.2f} "
            f"cf_faith_pearl_hard={rec['cf_faith_pearl_hard']:.2f}"
        )

    write_csv(res_dir / "per_instance.csv", all_instance_rows)
    append_table(TABLES_DIR / "table_axis_c_cf_faith.csv", table_rows)

    def _load_if_exists(path: Path):
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
        return None

    attribution = _load_if_exists(res_dir / "attribution.json")
    axis_a = _load_if_exists(res_dir / "axis_a_attribution.json")
    shift = _load_if_exists(res_dir / "shift_vr.json")
    axis_b = _load_if_exists(RESULTS_DIR / cfg.name / "axis_b_benchmark.json")

    results = {
        "provenance": {"config": cfg.as_dict(), "seed": cfg.seed, "n_cf": int(len(X_sel))},
        "methods": summary,           # Axis C + CF-faith, per CF method
        "attribution": attribution,   # IG deletion/insertion-AUC foil
        "axis_a": axis_a,             # Axis A: attribution causal-relevance (Phase 03)
        "axis_b": axis_b,             # Axis B: dataset graph diagnostic (Phase 01)
        "shift_vr": shift,            # Axis D: CF-method validity retention (Phase 03)
    }
    dump_json(res_dir / "summary.json", results)

    print()
    print_summary_table(summary)
    print(f"\n[04] wrote {res_dir / 'summary.json'}")
    print(f"[04] wrote {res_dir / 'per_instance.csv'}")
    print(f"[04] appended {TABLES_DIR / 'table_axis_c_cf_faith.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Axis-C + CF-faith on persisted CF arrays.")
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--seed", type=int, default=None,
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
