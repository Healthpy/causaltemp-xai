"""End-to-end CausalTemp-XAI benchmark harness.

Loads a persisted LinearSCM-T dataset and its frozen LSTM checkpoint, runs
Wachter, CARLA, and cfts-backed CF methods (COMTE, CONFETTI, CounTS, CELS)
over a subset of the test set, scores each with the full Axis-C + both
CF-faith metric suites, adds the Integrated-Gradients attribution foil and
the Shift-VR-lite robustness metric, then writes results.json and per_instance.csv.

Usage
-----
    uv run python experiments/run_all.py --config smoke           # CI smoke
    uv run python experiments/run_all.py --config full --n-cf 100  # paper results
    uv run python experiments/run_all.py --config smoke_nl        # NlinearSCM-T

For nonlinear configs (smoke_nl / full_nl, mechanism_type='mlp') the run
routes to a dedicated oracle-structural-CF path — no LSTM checkpoint and no
real CF methods are needed. See run_nonlinear().

Outputs (under experiments/):
    results.json         one summary record per method + attribution + shift-vr
    per_instance.csv     one row per (method, instance) for scatter / violins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual  # noqa: E402
from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import get_config, shifted_config  # noqa: E402
from causaltemp_xai.data_io import (  # noqa: E402
    DEFAULT_OUT_DIR,
    generate_and_save,
    load_dataset,
)
from causaltemp_xai.eval import evaluate_method, shift_vr  # noqa: E402
from causaltemp_xai.methods import CARLARecourse
from causaltemp_xai.methods import (  # noqa: E402
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
)
from causaltemp_xai.methods.counterfactual.cfts_methods import _DatasetAdapter  # noqa: E402
from causaltemp_xai.methods.attribution import (  # noqa: E402
    deletion_curve,
    insertion_curve,
    integrated_gradients,
)
from causaltemp_xai.scm.intervention import derive_intervention_t  # noqa: E402
from causaltemp_xai.metrics.axis_c import proximity, sparsity  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402

EXP_DIR = ROOT / "experiments"
TARGET_CLASS = 1


# ---------------------------------------------------------------------------
# CF-method plumbing
# ---------------------------------------------------------------------------


def _generate(method, X, clf, graph, mech) -> np.ndarray:
    """Generate one CF per instance, routing graph/mechanisms to causal methods."""
    import inspect
    params = inspect.signature(method.generate_batch).parameters
    if "graph" in params or "mechanism" in params:
        cfs = method.generate_batch(X, clf, graph, mech)
    else:
        cfs = method.generate_batch(X, clf)
    return np.asarray(cfs, dtype=np.float32)


def build_methods(X_train: np.ndarray) -> dict:
    """Construct native CF methods (Wachter + CARLA)."""
    return {
         "CARLA": CARLARecourse(
            target_class=TARGET_CLASS, n_steps=300, t0_fractions=(0.25, 0.5)
        ),
    }


def build_cfts_methods(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Construct cfts-backed CF methods with a shared training dataset."""
    ds = _DatasetAdapter(X_train, y_train)
    return {
        "CftsWachter": CftsWachterCF(target_class=TARGET_CLASS, dataset=ds, max_cfs=500),
        "CftsCOMTE":   CftsCOMTECF(target_class=TARGET_CLASS, dataset=ds),
        "CftsConfeti": CftsConfetiCF(target_class=TARGET_CLASS, dataset=ds),
        "CftsCounts":  CftsCountsCF(target_class=TARGET_CLASS, dataset=ds),
        "CftsCels":    CftsCelsCF(target_class=TARGET_CLASS, dataset=ds),
    }


def select_flip_candidates(clf, X_test, n_cf, target_class=TARGET_CLASS):
    """Indices of test instances the classifier predicts as NOT target_class."""
    preds = clf.predict(X_test)
    src = [i for i in range(len(X_test)) if preds[i] != target_class]
    if len(src) < n_cf:
        src = list(range(len(X_test)))
    return np.asarray(src[:n_cf], dtype=int)


# ---------------------------------------------------------------------------
# Per-instance metric dump
# ---------------------------------------------------------------------------


def per_instance_records(clf, X_sel, CFs, graph, mech, method_name):
    """One record per instance with per-CF metrics (for figures.py)."""
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    preds = np.asarray(clf.predict(CFs)).reshape(-1)
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mech)
        p = pearl.score(x, x_cf, t, graph, mech)
        rows.append({
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
        })
    return rows


# ---------------------------------------------------------------------------
# Attribution foil (Integrated Gradients + deletion/insertion curves)
# ---------------------------------------------------------------------------


def attribution_block(clf, X_sel, ig_steps=64, curve_steps=50):
    """Mean deletion/insertion AUC of IG saliency over the selected instances."""
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
        "insertion_minus_deletion": float(np.mean(ins_aucs) - np.mean(del_aucs)),
        "n": int(len(X_sel)),
    }


# ---------------------------------------------------------------------------
# Shift-VR environment
# ---------------------------------------------------------------------------


def load_or_make_shift_test(cfg, out_dir):
    """Return the shifted-environment test split (regenerated deterministically)."""
    shift_cfg = shifted_config(cfg, noise_type="uniform")
    dest = Path(out_dir) / shift_cfg.name
    if not (dest / "meta.json").exists():
        generate_and_save(shift_cfg, out_dir=out_dir)
    data = load_dataset(shift_cfg.name, out_dir=out_dir)
    return data["X_test"]


# ---------------------------------------------------------------------------
# NlinearSCM-T smoke path — oracle structural-CF positive control
# ---------------------------------------------------------------------------

ORACLE_SHIFT = 1.5


def build_oracle_cfs(X_sel, mechanism, noiseless, shift=ORACLE_SHIFT):
    """Build a (N, T, k) batch of oracle structural counterfactuals."""
    X_sel = np.asarray(X_sel, dtype=float)
    T = X_sel.shape[1]
    k = mechanism.k
    t0 = T // 2
    cfs = []
    for i, x in enumerate(X_sel):
        node = i % k
        value = float(x[t0, node]) + shift
        cfs.append(
            structural_counterfactual(x, mechanism, t0, node, value, noiseless=noiseless)
        )
    return np.asarray(cfs, dtype=np.float32)


def _score_oracle_batch(X_sel, CFs, graph, mechanism, method_name):
    """Classifier-free batch metrics for an oracle-CF method."""
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    prox_l1, prox_l2, spars = [], [], []
    r_h, r_s, p_h, p_s = [], [], [], []
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mechanism)
        p = pearl.score(x, x_cf, t, graph, mechanism)
        px1 = proximity(x, x_cf, norm="l1")
        px2 = proximity(x, x_cf, norm="l2")
        sp = sparsity(x, x_cf)
        prox_l1.append(px1); prox_l2.append(px2); spars.append(sp)
        r_h.append(r["hard"]); r_s.append(r["soft"])
        p_h.append(p["hard"]); p_s.append(p["soft"])
        rows.append({
            "method": method_name, "instance": int(i), "validity": None,
            "proximity_l1": px1, "proximity_l2": px2, "sparsity": sp,
            "intervention_t": int(t),
            "cf_faith_rollout_hard": r["hard"], "cf_faith_rollout_soft": r["soft"],
            "cf_faith_pearl_hard": p["hard"], "cf_faith_pearl_soft": p["soft"],
        })
    sparsity_mean = float(np.mean(spars))
    rec = {
        "method": method_name, "n": int(len(CFs)), "validity": None,
        "proximity_l1": float(np.mean(prox_l1)), "proximity_l2": float(np.mean(prox_l2)),
        "sparsity": sparsity_mean, "frac_altered": float(1.0 - sparsity_mean), "ood": None,
        "cf_faith_rollout_hard": float(np.mean(r_h)), "cf_faith_rollout_soft": float(np.mean(r_s)),
        "cf_faith_pearl_hard": float(np.mean(p_h)), "cf_faith_pearl_soft": float(np.mean(p_s)),
    }
    return rec, rows


def _ensure_dataset(config_name, out_dir):
    cfg = get_config(config_name)
    try:
        return load_dataset(cfg.name, out_dir=out_dir)
    except FileNotFoundError:
        print(f"[run_all] dataset for '{cfg.name}' missing — generating …")
        generate_and_save(cfg, out_dir=out_dir)
        return load_dataset(cfg.name, out_dir=out_dir)


def run_nonlinear(config_name, n_cf, out_dir):
    """NlinearSCM-T smoke run: score CF-faith on oracle structural-CFs."""
    cfg = get_config(config_name)
    data = _ensure_dataset(config_name, out_dir)
    X_test = data["X_test"]
    graph, mech = data["graph"], data["mechanism"]

    n = min(n_cf, len(X_test))
    X_sel = X_test[:n]
    print(
        f"[run_all] config={cfg.name} (nonlinear MLP) k={cfg.k} T={cfg.T} | "
        f"scoring CF-faith on {len(X_sel)} oracle structural-CFs"
    )

    summary, all_rows = [], []
    for name, noiseless in [("OracleCF-Pearl", False), ("OracleCF-Rollout", True)]:
        print(f"[run_all] building oracle CFs: {name} …")
        cfs = build_oracle_cfs(X_sel, mech, noiseless=noiseless)
        rec, rows = _score_oracle_batch(X_sel, cfs, graph, mech, name)
        summary.append(rec)
        all_rows.extend(rows)

    results = {
        "provenance": {
            "config": cfg.as_dict(), "seed": cfg.seed, "n_cf": int(len(X_sel)),
            "mechanism_type": cfg.mechanism_type, "target_class": TARGET_CLASS,
        },
        "methods": summary,
    }
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    per_instance_path = EXP_DIR / "per_instance.csv"
    _write_csv(per_instance_path, all_rows)
    _print_table(summary, attribution=None, shift=None)
    print(f"\n[run_all] wrote {results_path}")
    return results


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(config_name, n_cf, out_dir):
    cfg = get_config(config_name)
    if cfg.mechanism_type != "linear":
        return run_nonlinear(config_name, n_cf, out_dir)

    data = _ensure_dataset(config_name, out_dir)
    X_train, y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    graph, mech = data["graph"], data["mechanism"]

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt}; train first: "
            f"uv run python -m causaltemp_xai.classifiers.lstm --config {cfg.name} --train"
        )
    clf = LSTMClassifier.load(ckpt)

    accuracies = {
        "train": clf.score(X_train, data["Y_train"]),
        "val":   clf.score(data["X_val"], data["Y_val"]),
        "test":  clf.score(X_test, Y_test),
    }

    sel = select_flip_candidates(clf, X_test, n_cf)
    X_sel = X_test[sel]
    print(
        f"[run_all] config={cfg.name} k={cfg.k} T={cfg.T} | "
        f"test_acc={accuracies['test']:.3f} | "
        f"selected {len(X_sel)} flip candidates (target={TARGET_CLASS})"
    )

    # Native methods:  CARLA
    native_methods = build_methods(X_train)
    # cfts-backed methods
    cfts_methods = build_cfts_methods(X_train, y_train)
    all_methods = {**native_methods, **cfts_methods}

    summary, all_rows = [], []
    cf_cache = {}
    for name, method in all_methods.items():
        print(f"[run_all] generating CFs: {name} …")
        try:
            cfs = _generate(method, X_sel, clf, graph, mech)
            cf_cache[name] = cfs
            rec = evaluate_method(clf, X_sel, cfs, X_train, graph, mech, TARGET_CLASS)
            rec["method"] = name
            summary.append(rec)
            all_rows.extend(per_instance_records(clf, X_sel, cfs, graph, mech, name))
            print(f"[run_all]   {name}: validity={rec['validity']:.2f}  "
                  f"prox_l1={rec['proximity_l1']:.3f}  "
                  f"cf_faith_rollout_hard={rec['cf_faith_rollout_hard']:.2f}")
        except Exception as exc:
            print(f"[run_all]   {name} FAILED: {exc}")

    print("[run_all] integrated-gradients attribution foil …")
    attribution = attribution_block(clf, X_sel)

    print("[run_all] shift-VR-lite …")
    X_shift_test = load_or_make_shift_test(cfg, out_dir)
    shift_sel = select_flip_candidates(clf, X_shift_test, n_cf)
    X_shift_sel = X_shift_test[shift_sel]
    shift_methods = {k: v for k, v in native_methods.items()}  # shift-VR on native only
    shift = shift_vr(clf, shift_methods, X_sel, X_shift_sel, graph, mech, TARGET_CLASS)

    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "seed": cfg.seed,
            "n_cf": int(len(X_sel)),
            "classifier_accuracy": accuracies,
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


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
        f"{'Method':<16}{'valid':>7}{'prox_l1':>9}{'spars':>7}{'ood':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        v = f"{r['validity']:>7.2f}" if r["validity"] is not None else "    N/A"
        o = f"{r['ood']:>7.2f}" if r["ood"] is not None else "    N/A"
        print(
            f"{r['method']:<16}{v}{r['proximity_l1']:>9.3f}"
            f"{r['sparsity']:>7.2f}{o}"
            f"{r['cf_faith_rollout_hard']:>8.2f}{r['cf_faith_rollout_soft']:>8.2f}"
            f"{r['cf_faith_pearl_hard']:>8.2f}{r['cf_faith_pearl_soft']:>8.2f}"
        )
    if attribution:
        print("\n=== Attribution foil (IG) ===")
        print(
            f"  deletion_auc={attribution['deletion_auc']:.3f}  "
            f"insertion_auc={attribution['insertion_auc']:.3f}  "
            f"gap={attribution['insertion_minus_deletion']:+.3f}"
        )
    if shift:
        print("\n=== Shift-VR-lite (validity retention) ===")
        for name, m in shift.items():
            print(
                f"  {name:<12} base={m['validity_base']:.2f} "
                f"shift={m['validity_shift']:.2f} shift_vr={m['shift_vr']:.3f}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="CausalTemp-XAI end-to-end harness.")
    parser.add_argument(
        "--config", default="smoke",
        choices=["smoke", "full", "full_sparse", "smoke_nl", "full_nl"],
    )
    parser.add_argument(
        "--n-cf", type=int, default=None,
        help="Number of test instances to explain (default: 20 smoke / 100 full).",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    n_cf = args.n_cf or (20 if args.config.startswith("smoke") else 100)
    run(args.config, n_cf, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
