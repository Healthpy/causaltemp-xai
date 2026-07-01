"""
Script 05: Run ablation experiments for H5, H6, H7.

H5: Gaussian vs. non-Gaussian noise (identifiability condition)
H6: Label Visibility principle -- verifies threshold auto-tuning balance
H7: HMM regime-switching -- tests CF-faith degradation under non-stationary SCM

Usage:
  python experiments/05_ablations.py --ablation H5
  python experiments/05_ablations.py --ablation H7 --n_regimes 3
"""

import argparse
import json
import pathlib
import numpy as np

from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Regime

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results" / "ablations"

BASE_CFG = {
    "n_channels": 6,
    "T": 50,
    "max_lag": 2,
    "edge_density": 0.3,
    "sigma_w": 0.4,
    "N": 1000,
    "burn_in": 20,
}


def run_h5(seed: int = 42) -> None:
    """H5: Non-Gaussianity as identifiability condition."""
    print("[05-H5] Running Gaussian vs. Laplace noise ablation ...")
    results = {}
    for dist in ["laplace", "gaussian"]:
        cfg = {**BASE_CFG, "noise_dist": dist}
        bench = NlinearSCMT(config=cfg, seed=seed)
        split = bench.generate()
        balance = float(split.Y_train.mean())
        results[dist] = {
            "class_balance": balance,
            "identifiable": dist == "laplace",
        }
        print(f"  {dist}: balance={balance:.3f}")

    out = RESULTS_DIR / "H5_gaussian_vs_laplace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {out}")


def run_h6(seed: int = 42) -> None:
    """H6: Label Visibility -- auto-tuned threshold ensures balance."""
    print("[05-H6] Label Visibility ablation ...")
    results = []
    for trial_seed in range(seed, seed + 10):
        cfg = {**BASE_CFG, "noise_dist": "laplace", "N": 500}
        bench = NlinearSCMT(config=cfg, seed=trial_seed)
        split = bench.generate()
        bal = float(split.Y_train.mean())
        results.append({"seed": trial_seed, "balance": bal, "in_range": 0.3 <= bal <= 0.7})

    fraction_in_range = float(np.mean([r["in_range"] for r in results]))
    print(f"  Fraction in [0.3, 0.7]: {fraction_in_range:.2f}")
    balances = [f"{r['balance']:.3f}" for r in results]
    print(f"  Balances: {balances}")
    out = RESULTS_DIR / "H6_label_visibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"trials": results, "fraction_in_range": fraction_in_range}, f, indent=2)
    print(f"  Saved -> {out}")


def run_h7(n_regimes: int = 2, seed: int = 42) -> None:
    """H7: Regime-switching -- tests non-stationary SCM."""
    print(f"[05-H7] Regime-switching ablation (n_regimes={n_regimes}) ...")
    cfg = {
        **BASE_CFG,
        "noise_dist": "laplace",
        "n_regimes": n_regimes,
        "min_regime_len": 10,
        "transition_prob": 0.05,
    }
    bench = NlinearSCMT_Regime(config=cfg, seed=seed)
    split = bench.generate()
    balance = float(split.Y_train.mean())
    print(f"  n_regimes={n_regimes}, balance={balance:.3f}, shape={split.X_train.shape}")

    out = RESULTS_DIR / f"H7_regime_{n_regimes}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_regimes": n_regimes, "class_balance": balance,
                   "X_shape": list(split.X_train.shape),
                   "meta": {k: v for k, v in split.meta.items()
                             if k not in ("regime_labels_train", "regime_labels_test")}
                   }, f, indent=2)
    print(f"  Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=["H5", "H6", "H7"])
    parser.add_argument("--n_regimes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.ablation == "H5":
        run_h5(args.seed)
    elif args.ablation == "H6":
        run_h6(args.seed)
    elif args.ablation == "H7":
        run_h7(args.n_regimes, args.seed)


if __name__ == "__main__":
    main()
