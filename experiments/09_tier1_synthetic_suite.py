"""Phase 09: Tier-1 orchestration suite (M4i, `DECISIONS.md` 2026-08-06).

Runs the full/smoke experiment sequence (phases 01-04, then both
causal-discovery methods) across **all 3 synthetic families** -- linear (VAR),
nonlinear (MLP), spring -- so "which discovery method recovers which kind of
data best, compared against ground truth" is answerable from one file rather
than manually assembling per-config commands.

The kuramoto family was removed 2026-08-11 (`DECISIONS.md`); three synthetic
families are enough, and it was the worst-conditioned of the four.

Family -> preset table (spring has no full-scale preset yet):

    linear    mechanism_type="linear"    smoke -> "smoke"       full -> "full"
    mlp       mechanism_type="mlp"       smoke -> "smoke_nl"    full -> "full_nl"
    spring    mechanism_type="spring"    smoke -> "smoke_spring" full -> none

Per `(family, scale)` config, per seed:

1. Phases 01->04 via `experiments.08_aggregate_and_report.run_seed` (the
   established multi-run reuse point -- no reimplementation of the phase
   sequence).
2. `dynotears`/`pcmciplus` (`experiments.07_auxiliary_methods.run_graph_method`,
   the CF-faith graph-error decomposition) -- **only** for `mlp`/`spring`.
   `linear` is skipped and the reason recorded: `LinearMechanism`
   has no masking support (`build_masked_mechanism`), so the decomposition is
   genuinely not computable, not merely omitted.
3. `cross_method_agreement` (`run_cross_method_agreement`) -- **all 3
   families, including linear**. This is the one behavior change M4i makes to
   existing code: the function's `_MASKABLE` guard was dropped 2026-08-06
   after confirming its body never calls `build_masked_mechanism` (it only
   fits both methods and compares raw adjacencies via `shd`/`graph_auc`, both
   mechanism-type-agnostic). This is what answers "which method is suitable
   for which data" for `linear` too.

Usage
-----
    uv run python experiments/09_tier1_synthetic_suite.py --scale smoke --seeds 0
    uv run python experiments/09_tier1_synthetic_suite.py --scale both --families mlp spring
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR  # noqa: E402
from causaltemp_xai.stats import bootstrap_ci  # noqa: E402
from experiments._common import (  # noqa: E402
    RESULTS_DIR,
    TABLES_DIR,
    aggregate_across_seeds,
    dump_json,
    write_csv,
)

# Numbered phase modules are not valid dotted-import identifiers via a plain
# `import` (leading digit) -- importlib handles them fine, reusing the exact
# same functions the CLI entry points call (matches 08's own established
# pattern for cross-phase reuse).
_phase07 = importlib.import_module("experiments.07_auxiliary_methods")
_phase08 = importlib.import_module("experiments.08_aggregate_and_report")

_FAMILY_CONFIGS = {
    "linear": {"smoke": "smoke", "full": "full"},
    "mlp": {"smoke": "smoke_nl", "full": "full_nl"},
    "spring": {"smoke": "smoke_spring", "full": None},
}
_MASKABLE_FAMILIES = ("mlp", "spring")
ALL_FAMILIES = tuple(_FAMILY_CONFIGS)

_LINEAR_SKIP_REASON = (
    "LinearMechanism has no masking support (build_masked_mechanism); "
    "run_graph_method's CF-faith graph-error decomposition is genuinely not "
    "computable for linear, not merely omitted. cross_method_agreement (AUC/SHD "
    "vs ground truth) still runs for linear -- see below."
)

# Mirrors 08_aggregate_and_report.py's own argparse defaults exactly.
_DEFAULT_HPARAMS = dict(
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    lr=1e-3,
    batch_size=64,
    max_epochs=100,
    patience=10,
)


def _resolve_jobs(families: list[str], scales: list[str]) -> tuple[list, list]:
    """`(family, scale, config_name)` triples; a missing preset is skipped,
    printed, and recorded -- never silently dropped."""
    jobs, skipped = [], []
    for family in families:
        for scale in scales:
            config_name = _FAMILY_CONFIGS[family][scale]
            if config_name is None:
                reason = f"no {scale}-scale preset exists for family={family!r}"
                print(f"[09] skipping {family}/{scale}: {reason}")
                skipped.append({"family": family, "scale": scale, "reason": reason})
            else:
                jobs.append((family, scale, config_name))
    if not jobs:
        raise SystemExit(f"[09] no runnable jobs for families={families} scales={scales}")
    return jobs, skipped


def run_family_scale(
    family: str,
    scale: str,
    config_name: str,
    seeds: list[int],
    n_cf,
    out_dir,
    sweep: bool,
    ensemble_b: int,
    pc_alpha: float,
    n_boot: int,
    skip_runs: bool,
) -> list[dict]:
    """Run the full Tier-1 sequence for one `(family, scale)` config, across
    seeds. Returns one summary record per seed."""
    resolved_n_cf = n_cf or (20 if config_name.startswith("smoke") else 100)

    if not skip_runs:
        for seed in seeds:
            _phase08.run_seed(config_name, seed, resolved_n_cf, out_dir, _DEFAULT_HPARAMS)
    else:
        print(f"[09] --skip-runs: reusing existing per-seed results for {config_name!r}")

    rows = aggregate_across_seeds(config_name, seeds, classifier="lstm", n_boot=n_boot)
    write_csv(TABLES_DIR / f"table_seed_aggregate_{config_name}_lstm.csv", rows)

    per_seed_records = []
    for seed in seeds:
        record = {"family": family, "scale": scale, "config": config_name, "seed": seed}

        if family in _MASKABLE_FAMILIES:
            _phase07.run(
                config_name,
                out_dir,
                method="dynotears",
                n_cf=resolved_n_cf,
                sweep=sweep,
                seed=seed,
                ensemble_b=ensemble_b,
                pc_alpha=pc_alpha,
            )
            _phase07.run(
                config_name,
                out_dir,
                method="pcmciplus",
                n_cf=resolved_n_cf,
                sweep=sweep,
                seed=seed,
                ensemble_b=ensemble_b,
                pc_alpha=pc_alpha,
            )
            record["decomposition_skipped_reason"] = None
        else:
            record["decomposition_skipped_reason"] = _LINEAR_SKIP_REASON

        _phase07.run(
            config_name, out_dir, method="cross_method_agreement", seed=seed, pc_alpha=pc_alpha
        )

        # seeded_variant always renames (f"{base.name}_seed{seed}"), even for
        # seed == the base config's own seed -- read back from that path, not
        # the un-suffixed config_name. Call the real function rather than
        # duplicating its naming convention as a literal string.
        seeded_name = seeded_variant(get_config(config_name), seed).name
        cma_path = RESULTS_DIR / seeded_name / "cross_method_agreement" / "graph_agreement.json"
        with open(cma_path) as fh:
            cma = json.load(fh)
        record.update(
            {
                "dynotears_auc": cma["dynotears_auc"],
                "pcmciplus_auc": cma["pcmciplus_auc"],
                "shd_between_methods": cma["shd_between_methods"],
                "dynotears_shd_to_true": cma["dynotears_shd_to_true"],
                "pcmciplus_shd_to_true": cma["pcmciplus_shd_to_true"],
            }
        )
        per_seed_records.append(record)

    return per_seed_records


def _pool_suitability(records: list[dict]) -> list[dict]:
    """One row per `(family, scale, method)`, seed-pooled via `bootstrap_ci`
    (reused, not reimplemented)."""
    by_key: dict[tuple, dict] = {}
    for r in records:
        by_key.setdefault((r["family"], r["scale"]), []).append(r)

    rows = []
    for (family, scale), recs in by_key.items():
        for method, auc_key, shd_key in (
            ("dynotears", "dynotears_auc", "dynotears_shd_to_true"),
            ("pcmciplus", "pcmciplus_auc", "pcmciplus_shd_to_true"),
        ):
            aucs = [r[auc_key] for r in recs]
            shds = [r[shd_key] for r in recs]
            ci = bootstrap_ci(aucs, seed=0)
            rows.append(
                {
                    "family": family,
                    "scale": scale,
                    "method": method,
                    "auc_mean": ci.mean,
                    "auc_ci_lo": ci.ci_lo,
                    "auc_ci_hi": ci.ci_hi,
                    "shd_to_true_mean": sum(shds) / len(shds),
                    "mean_shd_between_methods": sum(r["shd_between_methods"] for r in recs)
                    / len(recs),
                    "n_seeds": len(recs),
                }
            )
    return rows


def run(
    families: list[str],
    scales: list[str],
    seeds: list[int],
    n_cf,
    out_dir,
    sweep: bool,
    ensemble_b: int,
    pc_alpha: float,
    n_boot: int,
    skip_runs: bool,
) -> None:
    jobs, skipped = _resolve_jobs(families, scales)

    all_records = []
    for family, scale, config_name in jobs:
        print(f"\n=== [09] {family}/{scale} -> config={config_name!r} ===")
        all_records.extend(
            run_family_scale(
                family,
                scale,
                config_name,
                seeds,
                n_cf,
                out_dir,
                sweep,
                ensemble_b,
                pc_alpha,
                n_boot,
                skip_runs,
            )
        )

    suitability = _pool_suitability(all_records)

    out_root = RESULTS_DIR / "tier1_suite"
    out_root.mkdir(parents=True, exist_ok=True)
    dump_json(
        out_root / "summary.json",
        {"per_seed_records": all_records, "skipped": skipped, "seeds": seeds},
    )
    write_csv(out_root / "table_method_suitability.csv", suitability)

    print(f"\n[09] wrote {out_root / 'summary.json'}")
    print(f"[09] wrote {out_root / 'table_method_suitability.csv'}")
    print()
    hdr = (
        f"{'family':<10}{'scale':<8}{'method':<12}{'AUC':>8}{'SHD_to_true':>14}{'SHD_between':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in suitability:
        print(
            f"{r['family']:<10}{r['scale']:<8}{r['method']:<12}{r['auc_mean']:>8.3f}"
            f"{r['shd_to_true_mean']:>14.1f}{r['mean_shd_between_methods']:>14.1f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scale", default="smoke", choices=("smoke", "full", "both"))
    parser.add_argument("--families", nargs="+", default=list(ALL_FAMILIES), choices=ALL_FAMILIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also compute the graph-quality sweep for maskable families (forwarded to "
        "run_graph_method --sweep). Off by default to keep the suite fast.",
    )
    parser.add_argument("--ensemble-b", type=int, default=0)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip phases 01-04 and discovery methods; only re-aggregate/re-pool "
        "already-produced results.",
    )
    args = parser.parse_args(argv)

    scales = ("smoke", "full") if args.scale == "both" else (args.scale,)
    run(
        args.families,
        list(scales),
        args.seeds,
        args.n_cf,
        args.out_dir,
        args.sweep,
        args.ensemble_b,
        args.pc_alpha,
        args.n_boot,
        args.skip_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
