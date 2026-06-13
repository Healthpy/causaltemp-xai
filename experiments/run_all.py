"""End-to-end CausalTemp-XAI benchmark harness.

Loads a persisted LinearSCM-T dataset and its frozen TCN checkpoint (Stages 2-3),
runs the three counterfactual methods (Wachter, DiCE, CARLA-causal) over a subset
of the test set, scores each with the full Axis-C + **both** CF-faith metric
suites (Stage 5), adds the Integrated-Gradients attribution foil (Stage 6) and the
Shift-VR-lite robustness metric (Stage 7), then writes ``results.json`` plus a
per-instance dump for the figures.

Usage
-----
    uv run python experiments/run_all.py --config smoke           # CI smoke
    uv run python experiments/run_all.py --config full --n-cf 100  # paper results

Outputs (under ``experiments/``):
    results.json            one summary record per method + attribution + shift-vr
                            + provenance (config, classifier accuracy, seed, n_cf)
    per_instance.csv        one row per (method, instance) for the scatter / violins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.attribution import (  # noqa: E402
    deletion_curve,
    insertion_curve,
    integrated_gradients,
)
from causaltemp_xai.classifiers import TCNClassifier  # noqa: E402
from causaltemp_xai.config import get_config, shifted_config  # noqa: E402
from causaltemp_xai.data_io import (  # noqa: E402
    DEFAULT_OUT_DIR,
    generate_and_save,
    load_dataset,
    stratified_split,
)
from causaltemp_xai.eval import evaluate_method, shift_vr  # noqa: E402
from causaltemp_xai.methods import CARLARecourse, DiCECF, WachterCF  # noqa: E402
from causaltemp_xai.methods.intervention import derive_intervention_t  # noqa: E402
from causaltemp_xai.metrics.axis_c import proximity, sparsity  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402

EXP_DIR = ROOT / "experiments"
TARGET_CLASS = 1


# ---------------------------------------------------------------------------
# CF-method plumbing
# ---------------------------------------------------------------------------


def _squeeze_cfs(cfs: np.ndarray) -> np.ndarray:
    """Collapse a ``(N, n_cfs, T, k)`` DiCE batch to ``(N, T, k)`` (first CF)."""
    cfs = np.asarray(cfs, dtype=np.float32)
    if cfs.ndim == 4:
        return cfs[:, 0]
    return cfs


def _generate(method, X, clf, graph, mech) -> np.ndarray:
    """Generate one CF per instance, routing graph/mechanisms to causal methods."""
    import inspect

    params = inspect.signature(method.generate_batch).parameters
    if "graph" in params or "mechanisms" in params:
        cfs = method.generate_batch(X, clf, graph, mech)
    else:
        cfs = method.generate_batch(X, clf)
    return _squeeze_cfs(cfs)


class _DiCESingle:
    """Adapter exposing ``generate_batch(X, model) -> (N, T, k)`` for shift_vr.

    ``shift_vr``/``validity`` expect a 3-D CF batch, but ``DiCECF`` returns a
    ``(N, n_cfs, T, k)`` diverse set. This wrapper keeps the uniform
    ``(X, model)`` signature (so ``eval._generate_batch`` routes it as a
    non-causal method) and returns the first CF per instance.
    """

    def __init__(self, dice: DiCECF) -> None:
        self._dice = dice

    def generate_batch(self, X, model):
        return _squeeze_cfs(self._dice.generate_batch(X, model))


def build_methods(X_train: np.ndarray, use_dice_ml: bool):
    """Construct the three CF methods. DiCE gets a (capped) background set."""
    background = X_train[: min(200, len(X_train))]
    wachter = WachterCF(target_class=TARGET_CLASS, n_steps=300, lr=0.1)
    dice = DiCECF(
        target_class=TARGET_CLASS,
        n_cfs=1,
        n_steps=300,
        background_data=background,
        use_dice_ml=use_dice_ml,
    )
    carla = CARLARecourse(
        target_class=TARGET_CLASS, n_steps=300, t0_fractions=(0.25, 0.5)
    )
    return {"Wachter": wachter, "DiCE": dice, "CARLA": carla}


def select_flip_candidates(clf, X_test, n_cf, target_class=TARGET_CLASS):
    """Indices of test instances the classifier predicts as NOT ``target_class``.

    Recourse for these is meaningful (there is a class to flip *to*). Falls back
    to the first ``n_cf`` instances if too few candidates exist.
    """
    preds = clf.predict(X_test)
    src = [i for i in range(len(X_test)) if preds[i] != target_class]
    if len(src) < n_cf:
        src = list(range(len(X_test)))
    return np.asarray(src[:n_cf], dtype=int)


# ---------------------------------------------------------------------------
# Per-instance metric dump (for the scatter / violins)
# ---------------------------------------------------------------------------


def per_instance_records(clf, X_sel, CFs, graph, mech, method_name):
    """One record per instance with per-cf metrics (for figures.py)."""
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    preds = np.asarray(clf.predict(CFs)).reshape(-1)
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mech)
        p = pearl.score(x, x_cf, t, graph, mech)
        rows.append(
            {
                "method": method_name,
                "instance": int(i),
                "validity": int(preds[i] == TARGET_CLASS),
                "proximity_l1": proximity(x, x_cf, norm="l1"),
                "proximity_l2": proximity(x, x_cf, norm="l2"),
                "sparsity": sparsity(x, x_cf),
                "intervention_t": int(t),
                "cf_faith_rollout_hard": r["hard"],
                "cf_faith_rollout_soft": r["soft"],
                "cf_faith_pearl_hard": p["hard"],
                "cf_faith_pearl_soft": p["soft"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Attribution foil (Integrated Gradients + deletion/insertion curves)
# ---------------------------------------------------------------------------


def attribution_block(clf, X_sel, ig_steps=64, curve_steps=50):
    """Mean deletion/insertion AUC of IG saliency over the selected instances.

    The foil: a faithful saliency method can score well here (sharp deletion
    drop / insertion rise) while telling us nothing about causal faithfulness.
    """
    del_aucs, ins_aucs = [], []
    for x in X_sel:
        ig = integrated_gradients(clf, x, TARGET_CLASS, steps=ig_steps)
        _, del_auc = deletion_curve(clf, x, ig, TARGET_CLASS, n_steps=curve_steps)
        _, ins_auc = insertion_curve(clf, x, ig, TARGET_CLASS, n_steps=curve_steps)
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
    return {
        "method": "IntegratedGradients",
        "deletion_auc": float(np.mean(del_aucs)),
        "insertion_auc": float(np.mean(ins_aucs)),
        # A good map: low deletion AUC, high insertion AUC → large positive gap.
        "insertion_minus_deletion": float(np.mean(ins_aucs) - np.mean(del_aucs)),
        "n": int(len(X_sel)),
    }


# ---------------------------------------------------------------------------
# Shift-VR environment
# ---------------------------------------------------------------------------


def load_or_make_shift_test(cfg, out_dir):
    """Return the shifted-environment test split (regenerated deterministically).

    Same SCM (graph/mechanisms) as the base config, ``noise_type='uniform'``.
    Persisted under ``<cfg>_shift`` so re-runs are cheap.
    """
    shift_cfg = shifted_config(cfg, noise_type="uniform")
    dest = Path(out_dir) / shift_cfg.name
    if not (dest / "meta.json").exists():
        generate_and_save(shift_cfg, out_dir=out_dir)
    data = load_dataset(shift_cfg.name, out_dir=out_dir)
    return data["X_test"]


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(config_name, n_cf, out_dir, use_dice_ml):
    cfg = get_config(config_name)
    data = load_dataset(cfg.name, out_dir=out_dir)
    X_train = data["X_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    graph, mech = data["graph"], data["mechanisms"]

    ckpt = Path(out_dir) / cfg.name / "tcn.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt}; train first: "
            f"uv run python -m causaltemp_xai.classifiers.tcn --config {cfg.name} --train"
        )
    clf = TCNClassifier.load(ckpt)

    accuracies = {
        "train": clf.score(X_train, data["Y_train"]),
        "val": clf.score(data["X_val"], data["Y_val"]),
        "test": clf.score(X_test, Y_test),
    }

    sel = select_flip_candidates(clf, X_test, n_cf)
    X_sel = X_test[sel]
    print(
        f"[run_all] config={cfg.name} k={cfg.k} T={cfg.T} | "
        f"test_acc={accuracies['test']:.3f} | "
        f"selected {len(X_sel)} flip candidates (target={TARGET_CLASS})"
    )

    methods = build_methods(X_train, use_dice_ml=use_dice_ml)

    summary, all_rows = [], []
    cf_cache = {}
    for name, method in methods.items():
        print(f"[run_all] generating CFs: {name} …")
        cfs = _generate(method, X_sel, clf, graph, mech)
        cf_cache[name] = cfs
        rec = evaluate_method(clf, X_sel, cfs, X_train, graph, mech, TARGET_CLASS)
        rec["method"] = name
        if name == "DiCE":
            rec["dice_backend"] = methods["DiCE"].backend_used
        summary.append(rec)
        all_rows.extend(per_instance_records(clf, X_sel, cfs, graph, mech, name))

    print("[run_all] integrated-gradients attribution foil …")
    attribution = attribution_block(clf, X_sel)

    print("[run_all] shift-VR-lite …")
    X_shift_test = load_or_make_shift_test(cfg, out_dir)
    shift_sel = select_flip_candidates(clf, X_shift_test, n_cf)
    X_shift_sel = X_shift_test[shift_sel]
    shift_methods = {
        "Wachter": methods["Wachter"],
        "DiCE": _DiCESingle(methods["DiCE"]),
        "CARLA": methods["CARLA"],
    }
    shift = shift_vr(clf, shift_methods, X_sel, X_shift_sel, graph, mech, TARGET_CLASS)

    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "seed": cfg.seed,
            "n_cf": int(len(X_sel)),
            "classifier_accuracy": accuracies,
            "dice_backend": methods["DiCE"].backend_used,
            "target_class": TARGET_CLASS,
        },
        "methods": summary,
        "attribution": attribution,
        "shift_vr": shift,
    }

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    per_instance_path = EXP_DIR / "per_instance.csv"
    _write_csv(per_instance_path, all_rows)

    _print_table(summary, attribution, shift)
    print(f"\n[run_all] wrote {results_path}")
    print(f"[run_all] wrote {per_instance_path}")
    return results


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _print_table(summary, attribution, shift):
    print("\n=== Axis-C + CF-faith (batch means) ===")
    hdr = (
        f"{'Method':<10}{'valid':>7}{'prox_l1':>9}{'spars':>7}{'ood':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        print(
            f"{r['method']:<10}{r['validity']:>7.2f}{r['proximity_l1']:>9.3f}"
            f"{r['sparsity']:>7.2f}{r['ood']:>7.2f}"
            f"{r['cf_faith_rollout_hard']:>8.2f}{r['cf_faith_rollout_soft']:>8.2f}"
            f"{r['cf_faith_pearl_hard']:>8.2f}{r['cf_faith_pearl_soft']:>8.2f}"
        )
    print("\n=== Attribution foil (IG) ===")
    print(
        f"  deletion_auc={attribution['deletion_auc']:.3f}  "
        f"insertion_auc={attribution['insertion_auc']:.3f}  "
        f"gap={attribution['insertion_minus_deletion']:+.3f}"
    )
    print("\n=== Shift-VR-lite (validity retention) ===")
    for name, m in shift.items():
        print(
            f"  {name:<10} base={m['validity_base']:.2f} "
            f"shift={m['validity_shift']:.2f} shift_vr={m['shift_vr']:.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="CausalTemp-XAI end-to-end harness.")
    parser.add_argument("--config", default="smoke", choices=["smoke", "full", "full_sparse"])
    parser.add_argument(
        "--n-cf",
        type=int,
        default=None,
        help="Number of test instances to explain (default: 20 smoke / 100 full).",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--no-dice-ml",
        action="store_true",
        help="Use the from-scratch DPP DiCE fallback instead of the dice-ml gradient backend.",
    )
    args = parser.parse_args(argv)

    n_cf = args.n_cf
    if n_cf is None:
        n_cf = 20 if args.config == "smoke" else 100

    run(args.config, n_cf, args.out_dir, use_dice_ml=not args.no_dice_ml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
