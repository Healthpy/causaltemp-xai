"""run_all.py – End-to-end benchmark experiment.

Generates synthetic time-series data from LinearSCM-T, trains a TCN
classifier, runs three counterfactual methods, and reports Axis-C and
CF-faith metrics.

Usage
-----
    python experiments/run_all.py [--n_samples 500] [--T 50] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on the path when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmark import LinearSCMT
from causaltemp_xai.classifiers import TCNClassifier
from causaltemp_xai.methods import WachterCF, DiCECF, CARLACF
from causaltemp_xai.metrics import (
    cf_faith_hard,
    cf_faith_soft,
    validity,
    proximity,
    sparsity,
    ood_plausibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predict_fn(clf: TCNClassifier):
    """Return a torch-compatible prediction callable from a TCNClassifier."""
    import torch

    def predict_fn(x_tensor):
        x_np = x_tensor.detach().cpu().numpy()
        proba = clf.predict_proba(x_np)
        return torch.tensor(proba, dtype=torch.float32)

    return predict_fn


def _evaluate_cfs(
    clf: TCNClassifier,
    X_test: np.ndarray,
    CFs: np.ndarray,
    X_train: np.ndarray,
    causal_mask: np.ndarray,
    target_class: int,
    method_name: str,
) -> dict:
    """Compute all metrics for a set of counterfactuals."""
    # Axis-C
    val = validity(clf.predict, CFs, target_class).mean()
    prox = proximity(X_test, CFs).mean()
    spar = sparsity(X_test, CFs).mean()
    ood = ood_plausibility(CFs, X_train).mean()

    # CF-faith (averaged over test set)
    faith_hard = np.mean([
        cf_faith_hard(X_test[i], CFs[i], causal_mask) for i in range(len(X_test))
    ])
    faith_soft = np.mean([
        cf_faith_soft(X_test[i], CFs[i], causal_mask) for i in range(len(X_test))
    ])

    results = {
        "method": method_name,
        "validity": float(val),
        "proximity_mean": float(prox),
        "sparsity_mean": float(spar),
        "ood_plausibility_mean": float(ood),
        "cf_faith_hard": float(faith_hard),
        "cf_faith_soft": float(faith_soft),
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    print(f"[run_all] seed={args.seed}, n_samples={args.n_samples}, T={args.T}")

    # ---- Data generation ----
    generator = LinearSCMT(
        n_vars=args.n_vars,
        lag=args.lag,
        seed=args.seed,
    )
    X, y = generator.sample_with_labels(
        n_samples=args.n_samples,
        T=args.T,
        target_var=0,
    )

    # Causal mask: only first variable (and all its lags) are "causal"
    causal_mask = np.zeros((args.T, args.n_vars), dtype=bool)
    causal_mask[:, 0] = True

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[run_all] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[run_all] Class distribution (train): {np.bincount(y_train)}")

    # Save generated data
    data_dir = ROOT / "data" / "linearscm_t"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "X_test.npy", X_test)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "y_test.npy", y_test)

    # ---- Classifier ----
    clf = TCNClassifier(
        n_vars=args.n_vars,
        n_classes=2,
        n_epochs=args.n_epochs,
    )
    clf.fit(X_train, y_train)
    train_acc = (clf.predict(X_train) == y_train).mean()
    test_acc = (clf.predict(X_test) == y_test).mean()
    print(f"[run_all] TCN accuracy — train: {train_acc:.3f}, test: {test_acc:.3f}")

    predict_fn = _make_predict_fn(clf)

    # Select a small subset for CF generation (expensive)
    n_cf = min(args.n_cf, len(X_test))
    X_cf_input = X_test[:n_cf]
    target_class = 1  # always explain as class 1

    all_results = []

    # ---- Wachter ----
    print("[run_all] Running Wachter…")
    wachter = WachterCF(predict_fn, target_class=target_class)
    CFs_wachter = wachter.generate_batch(X_cf_input)
    all_results.append(
        _evaluate_cfs(clf, X_cf_input, CFs_wachter, X_train, causal_mask, target_class, "Wachter")
    )

    # ---- DiCE ----
    print("[run_all] Running DiCE…")
    dice = DiCECF(predict_fn, target_class=target_class, n_cfs=1)
    # DiCE returns (n_cf, n_cfs, T, d) — squeeze the n_cfs=1 dimension
    CFs_dice_raw = dice.generate_batch(X_cf_input)
    CFs_dice = CFs_dice_raw[:, 0]
    all_results.append(
        _evaluate_cfs(clf, X_cf_input, CFs_dice, X_train, causal_mask, target_class, "DiCE")
    )

    # ---- CARLA ----
    print("[run_all] Running CARLA…")
    carla = CARLACF(predict_fn, target_class=target_class)
    CFs_carla = carla.generate_batch(X_cf_input)
    all_results.append(
        _evaluate_cfs(clf, X_cf_input, CFs_carla, X_train, causal_mask, target_class, "CARLA")
    )

    # ---- Report ----
    print("\n=== Results ===")
    header = f"{'Method':<10} {'Validity':>9} {'Proximity':>10} {'Sparsity':>9} {'OOD':>8} {'Faith-H':>8} {'Faith-S':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['method']:<10} "
            f"{r['validity']:>9.3f} "
            f"{r['proximity_mean']:>10.4f} "
            f"{r['sparsity_mean']:>9.3f} "
            f"{r['ood_plausibility_mean']:>8.4f} "
            f"{r['cf_faith_hard']:>8.3f} "
            f"{r['cf_faith_soft']:>8.3f}"
        )

    # Save results
    results_path = ROOT / "experiments" / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[run_all] Results saved to {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="causaltemp-xai benchmark")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n_vars", type=int, default=5)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--n_cf", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
