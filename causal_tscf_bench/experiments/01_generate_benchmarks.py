"""
Script 01: Generate benchmark datasets from YAML configs.

Usage:
  python experiments/01_generate_benchmarks.py --config configs/linear_scm_t.yaml
  python experiments/01_generate_benchmarks.py --all
"""

import argparse
import pathlib
import yaml

from causal_tscf_bench.benchmarks.linear_scm_t import LinearSCMT
from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Nonmonotonic, NlinearSCMT_Regime

BENCHMARK_MAP = {
    "LinearSCMT": LinearSCMT,
    "NlinearSCMT": NlinearSCMT,
    "NlinearSCMT_Nonmonotonic": NlinearSCMT_Nonmonotonic,
    "NlinearSCMT_Regime": NlinearSCMT_Regime,
}

CONFIG_DIR = pathlib.Path(__file__).parent.parent / "configs"
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data"


def generate_from_config(cfg_path: pathlib.Path) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    bclass = BENCHMARK_MAP[cfg["benchmark"]]
    seed = cfg.pop("seed", 42)
    bench_name = cfg.pop("benchmark")

    bench = bclass(config=cfg, seed=seed)
    split = bench.generate()

    out_dir = OUTPUT_DIR / bench_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bench.save(str(out_dir))

    print(f"[01] {bench_name}")
    print(f"     Train {split.X_train.shape}  Val {split.X_val.shape}  Test {split.X_test.shape}")
    print(f"     Class balance (train): {split.Y_train.mean():.3f}")
    print(f"     Causal channel: {split.meta.get('causal_channel')}  "
          f"Label threshold: {split.meta.get('label_threshold', 'N/A'):.4f}")
    print(f"     Saved -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for cfg_path in sorted(CONFIG_DIR.glob("*.yaml")):
            print(f"\n--- {cfg_path.name} ---")
            generate_from_config(cfg_path)
    elif args.config:
        generate_from_config(pathlib.Path(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
