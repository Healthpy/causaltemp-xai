"""Run cfts-backed CF methods and score them with the full metric suite.

Methods:
    CftsWachterCF   -- gradient-based Wachter (cfts library)
    CftsCOMTECF     -- COMTE shapelet replacement (cfts library)
    CftsConfetiCF   -- genetic CONFETTI (cfts library)
    CftsCountsCF    -- VAE-based CounTS (cfts library)
    CftsCelsCF      -- saliency-guided CELS/M-CELS (cfts library)

Usage
-----
    uv run python experiments/run_cfts.py --config smoke
    uv run python experiments/run_cfts.py --config full --n-cf 100

Outputs (under ``experiments/``):
    cfts_results.json        summary per method
    cfts_per_instance.csv    per-(method, instance) metrics for figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import get_config  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.eval import evaluate_method  # noqa: E402
from causaltemp_xai.methods import (  # noqa: E402
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
)
from causaltemp_xai.methods.counterfactual.cfts_methods import _DatasetAdapter  # noqa: E402
from causaltemp_xai.scm.intervention import derive_intervention_t  # noqa: E402
from causaltemp_xai.metrics.axis_c import proximity, sparsity  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402

EXP_DIR = ROOT / "experiments"
TARGET_CLASS = 1


# ---------------------------------------------------------------------------
# Helpers (shared with run_all.py)
# ---------------------------------------------------------------------------


def select_flip_candidates(clf, X_test, n_cf, target_class=TARGET_CLASS):
    preds = clf.predict(X_test)
    src = [i for i in range(len(X_test)) if preds[i] != target_class]
    if len(src) < n_cf:
        src = list(range(len(X_test)))
    return np.asarray(src[:n_cf], dtype=int)


def per_instance_records(clf, X_sel, cfs, graph, mech, method_name):
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    preds = np.asarray(clf.predict(cfs)).reshape(-1)
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, cfs)):
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


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _print_table(summary):
    print("\n=== cfts methods: Axis-C + CF-faith (batch means) ===")
    hdr = (
        f"{'Method':<16}{'valid':>7}{'prox_l1':>9}{'spars':>7}{'ood':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        print(
            f"{r['method']:<16}{r['validity']:>7.2f}{r['proximity_l1']:>9.3f}"
            f"{r['sparsity']:>7.2f}{r['ood']:>7.2f}"
            f"{r['cf_faith_rollout_hard']:>8.2f}{r['cf_faith_rollout_soft']:>8.2f}"
            f"{r['cf_faith_pearl_hard']:>8.2f}{r['cf_faith_pearl_soft']:>8.2f}"
        )


# ---------------------------------------------------------------------------
# Build cfts methods
# ---------------------------------------------------------------------------


def build_cfts_methods(X_train, y_train):
    """Construct cfts CF methods with a shared training dataset."""
    ds = _DatasetAdapter(X_train, y_train)
    return {
        "CftsWachter": CftsWachterCF(target_class=TARGET_CLASS, dataset=ds, max_cfs=500),
        "CftsCOMTE": CftsCOMTECF(target_class=TARGET_CLASS, dataset=ds),
        "CftsConfeti": CftsConfetiCF(target_class=TARGET_CLASS, dataset=ds),
        "CftsCounts": CftsCountsCF(target_class=TARGET_CLASS, dataset=ds),
        "CftsCels": CftsCelsCF(target_class=TARGET_CLASS, dataset=ds),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(config_name, n_cf, out_dir):
    cfg = get_config(config_name)
    data = load_dataset(cfg.name, out_dir=out_dir)
    X_train, y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    graph, mech = data["graph"], data["mechanism"]

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Train first:\n"
            f"  uv run python -m causaltemp_xai.classifiers.lstm --config {cfg.name} --train"
        )
    clf = LSTMClassifier.load(ckpt)

    test_acc = clf.score(X_test, Y_test)
    sel = select_flip_candidates(clf, X_test, n_cf)
    X_sel = X_test[sel]
    print(
        f"[run_cfts] config={cfg.name} k={cfg.k} T={cfg.T} | "
        f"test_acc={test_acc:.3f} | {len(X_sel)} flip candidates"
    )

    methods = build_cfts_methods(X_train, y_train)

    summary, all_rows = [], []
    for name, method in methods.items():
        print(f"[run_cfts] generating CFs: {name} ...")
        try:
            cfs = method.generate_batch(X_sel, clf)
            cfs = np.asarray(cfs, dtype=np.float32)
            rec = evaluate_method(clf, X_sel, cfs, X_train, graph, mech, TARGET_CLASS)
            rec["method"] = name
            summary.append(rec)
            all_rows.extend(per_instance_records(clf, X_sel, cfs, graph, mech, name))
            print(f"[run_cfts]   {name}: validity={rec['validity']:.2f} prox_l1={rec['proximity_l1']:.3f}")
        except Exception as exc:
            print(f"[run_cfts]   {name} FAILED: {exc}")

    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "n_cf": int(len(X_sel)),
            "test_acc": float(test_acc),
            "target_class": TARGET_CLASS,
        },
        "methods": summary,
    }

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_json = EXP_DIR / "cfts_results.json"
    out_csv = EXP_DIR / "cfts_per_instance.csv"

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    _write_csv(out_csv, all_rows)

    _print_table(summary)
    print(f"\n[run_cfts] wrote {out_json}")
    print(f"[run_cfts] wrote {out_csv}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run cfts CF methods on a saved dataset/classifier.")
    parser.add_argument("--config", default="smoke", choices=["smoke", "full", "full_sparse"])
    parser.add_argument("--n-cf", type=int, default=None,
                        help="Number of test instances to explain (default: 20 smoke / 100 full).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    n_cf = args.n_cf or (20 if args.config == "smoke" else 100)
    run(args.config, n_cf, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
