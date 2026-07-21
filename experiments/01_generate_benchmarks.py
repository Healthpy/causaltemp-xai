"""Phase 01: Generate and persist SCM-T benchmark datasets.

Thin CLI wrapper around ``causaltemp_xai.data_io.generate_and_save`` that can
generate a single config or every registered config in one call. Datasets are
deterministic (seeded), so they are written under ``data/scm_t/<config>/``
(gitignored, regenerated on demand) rather than into ``results/``.

Also computes the **Axis B** graph diagnostic (SHD/LagAcc/TV-Confounding) once
per dataset -- see ``experiments/_common.py`` module docstring for why Axis B
is a dataset-level diagnostic rather than a per-method score in this pipeline
(no graph-discovery method is wired in) -- and writes it to
``results/<config>/axis_b_benchmark.json``.

Usage
-----
    uv run python experiments/01_generate_benchmarks.py --config smoke
    uv run python experiments/01_generate_benchmarks.py --all
    uv run python experiments/01_generate_benchmarks.py --config smoke --shift-noise uniform
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import CONFIGS, get_config, seeded_variant, shifted_config  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, generate_and_save, load_dataset  # noqa: E402
from experiments._common import RESULTS_DIR, axis_b_benchmark_diagnostic, dump_json  # noqa: E402


def generate_one(config_name: str, out_dir, shift_noise: str | None, seed: int | None = None) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    if shift_noise is not None:
        cfg = shifted_config(cfg, noise_type=shift_noise)

    dest = generate_and_save(cfg, out_dir=out_dir)
    with open(dest / "meta.json") as fh:
        meta = json.load(fh)
    print(f"[01] {cfg.name} ({cfg.mechanism_type}) -> {dest}")
    print(f"     split sizes:   {meta['split_sizes']}")
    print(f"     class balance: {meta['class_balance']}")

    data = load_dataset(cfg.name, out_dir=out_dir)
    axis_b = axis_b_benchmark_diagnostic(
        data["graph"], data["X_test"], mechanism=data.get("mechanism")
    )
    axis_b_path = RESULTS_DIR / cfg.name / "axis_b_benchmark.json"
    dump_json(axis_b_path, axis_b)
    print(
        f"     axis B (graph diagnostic): SHD={axis_b['SHD']:.0f} "
        f"LagAcc={axis_b['LagAcc']:.2f} ResidualDep={axis_b.get('ResidualDep', float('nan')):.3f}"
    )
    print(f"     -> {axis_b_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate SCM-T benchmark datasets.")
    parser.add_argument("--config", choices=sorted(CONFIGS), default=None)
    parser.add_argument("--all", action="store_true", help="Generate every registered config.")
    parser.add_argument(
        "--shift-noise", choices=("laplace", "uniform"), default=None,
        help="Also/instead generate the Shift-VR environment for --config (same SCM, different noise).",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Multi-seed replicate (M2): override --config's registered seed "
            "via causaltemp_xai.config.seeded_variant, writing under "
            "'<config>_seed<seed>' instead of '<config>'. Ignored with --all."
        ),
    )
    args = parser.parse_args(argv)

    if not args.all and args.config is None:
        parser.error("pass --config <name> or --all")

    names = sorted(CONFIGS) if args.all else [args.config]
    for name in names:
        generate_one(
            name, args.out_dir, args.shift_noise if not args.all else None,
            seed=args.seed if not args.all else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
