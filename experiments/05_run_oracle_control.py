"""Phase 05: Oracle structural-CF **positive control** — any mechanism.

This phase runs **no explainer and no classifier**. It is the oracle
counterpart to Phase 03/04, not their nonlinear variant: where 03 runs real CF
methods (Wachter/CARLA/cfts-*) against a trained LSTM and 04 scores what they
produced, this phase *constructs* counterfactuals analytically via
:func:`~causaltemp_xai.benchmarks.structural_cf.structural_counterfactual`
(abduct -> intervene -> re-roll) and scores CF-faith **classifier-free** on
them.

Its job is the positive control the rest of Axis C leans on: these CFs are
correct **by construction**, so CF-faith *must* certify them. Without that
anchor, a low CF-faith across every real method is ambiguous — broken methods
and a broken metric look identical. Two mutually-exclusive variants are
emitted, which also pins the rollout-vs-Pearl contrast at experiment scale:
``OracleCF-Pearl`` (noisy, built to satisfy ``pearl_delta``) and
``OracleCF-Rollout`` (noiseless, built to satisfy ``noiseless_rollout``).

**Runs on every config, linear and nonlinear** (2026-07-15). It was previously
named ``05_run_oracle_nonlinear.py`` and hard-rejected linear configs with
"use 03 + 04 instead" — stale advice from when nonlinear CF methods were
deferred to a collaborator's track (``docs/plans/nlinearscm-t/index.md``
Backlog #2; Phase 03 has since supported ``*_nl`` configs). That gate conflated
*oracle vs explainer* with *linear vs nonlinear*: 03+04 give you explainers on
linear, never the oracle control, so the linear configs had no experiment-scale
control at all (only unit-scale, in ``tests/test_metric_adversarial.py``).
``build_oracle_cfs`` is mechanism-generic — ``structural_counterfactual`` routes
through ``mechanism.forward_numpy``, which ``LinearMechanism`` and
``MLPMechanism`` both implement — so the gate protected nothing.

Outputs (under ``results/<config>/oracle/``)::

    eval_<Name>.json
    per_instance.csv
    summary.json

and appends aggregated rows to ``results/tables/table_axis_c_cf_faith.csv``.

Usage
-----
    uv run python experiments/05_run_oracle_control.py --config smoke
    uv run python experiments/05_run_oracle_control.py --config smoke_nl
    uv run python experiments/05_run_oracle_control.py --config full_nl --n-cf 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, generate_and_save, load_dataset  # noqa: E402
from experiments._common import (  # noqa: E402
    TABLES_DIR,
    aggregate_method_row,
    append_table,
    config_dir,
    dump_json,
    per_instance_records,
    print_summary_table,
    set_run_context,
    write_csv,
)

ORACLE_SHIFT = 1.5


def build_oracle_cfs(X_sel, mechanism, noiseless, shift=ORACLE_SHIFT) -> np.ndarray:
    """Build a (N, T, k) batch of oracle structural counterfactuals."""
    X_sel = np.asarray(X_sel, dtype=float)
    T = X_sel.shape[1]
    k = mechanism.k
    t0 = T // 2
    cfs = []
    for i, x in enumerate(X_sel):
        node = i % k
        value = float(x[t0, node]) + shift
        cfs.append(structural_counterfactual(x, mechanism, t0, node, value, noiseless=noiseless))
    return np.asarray(cfs, dtype=np.float32)


def run(config_name: str, n_cf: int, out_dir) -> None:
    cfg = get_config(config_name)
    set_run_context(seed=cfg.seed, config=cfg.name)
    try:
        data = load_dataset(cfg.name, out_dir=out_dir)
    except FileNotFoundError:
        print(f"[05] dataset for '{cfg.name}' missing -- generating ...")
        generate_and_save(cfg, out_dir=out_dir)
        data = load_dataset(cfg.name, out_dir=out_dir)

    X_test = data["X_test"]
    graph, mech = data["graph"], data["mechanism"]
    n = min(n_cf, len(X_test))
    X_sel = X_test[:n]
    print(
        f"[05] config={cfg.name} ({cfg.mechanism_type}) k={cfg.k} T={cfg.T} | "
        f"scoring CF-faith on {len(X_sel)} oracle structural-CFs"
    )

    res_dir = config_dir(cfg.name, "oracle")
    summary, all_instance_rows, table_rows = [], [], []
    for name, noiseless in [("OracleCF-Pearl", False), ("OracleCF-Rollout", True)]:
        print(f"[05] building oracle CFs: {name} ...")
        cfs = build_oracle_cfs(X_sel, mech, noiseless=noiseless)

        instance_rows = per_instance_records(cfg.name, "oracle", name, X_sel, cfs, graph, mech)
        all_instance_rows.extend(instance_rows)
        rec = aggregate_method_row(cfg.name, "oracle", name, instance_rows)
        table_rows.append(rec)
        summary.append(rec)

        dump_json(res_dir / f"eval_{name}.json", rec)

    write_csv(res_dir / "per_instance.csv", all_instance_rows)
    append_table(TABLES_DIR / "table_axis_c_cf_faith.csv", table_rows)

    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "seed": cfg.seed,
            "n_cf": len(X_sel),
            "mechanism_type": cfg.mechanism_type,
        },
        "methods": summary,
    }
    dump_json(res_dir / "summary.json", results)

    print()
    print_summary_table(summary)
    print(f"\n[05] wrote {res_dir / 'summary.json'}")
    print(f"[05] appended {TABLES_DIR / 'table_axis_c_cf_faith.csv'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Oracle structural-CF positive control (any mechanism; classifier-free)."
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    n_cf = args.n_cf or (20 if args.config.startswith("smoke") else 100)
    run(args.config, n_cf, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
