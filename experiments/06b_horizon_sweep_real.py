"""Phase 06b: horizon sweep on real signal (M4b) -- no causal graph.

Real-data sibling of ``experiments/06_horizon_sweep.py``, testing whether
the horizon/decay result (H8: validity falls as intervention-to-outcome
distance ``T - t0`` grows) reproduces outside synthetic SCMs. **No causal
graph, no CF-faith, no PNS** -- those all need a known oracle mechanism,
which real data doesn't have (`ROADMAP.md` M4b DoD: "discovered-graph
CF-faith scoring is explicitly out").

**M4g addendum (`DECISIONS.md` 2026-08-06):** this phase now also persists
``X_sel.npy``/``X_cf_<Method>.npy`` under ``results/real_<name>/lstm/cf/``
(mirroring Phase 03's on-disk contract), which did not exist before this
change. That is purely a prerequisite for M4g's *separate* discovered-graph
CF-faith scorer (`experiments/07b_discovered_graph_real.py`) to read from --
this phase's own scope (no graph, no CF-faith) is unchanged.

**Why this is not a per-horizon sweep like Phase 06.** Phase 06 re-generates
CARLA-family CFs at each swept ``t0`` because ``CARLARecourse``/
``PearlCARLARecourse`` take ``t0`` as a constructor argument. Both need a
``mechanism`` and are therefore unusable here (`DECISIONS.md` 2026-08-05).
The 5 graph-free methods used instead (``CftsWachter``, ``CftsCOMTE``,
``CftsCounts``, ``CftsConfeti``, ``CftsCels``) have **no** ``t0`` parameter
at all -- they edit the trajectory freely and their intervention point is
purely a *derived* property of the CF they happen to produce (same "cfts
baselines have no intervention timestep to pin" point Phase 06's own
docstring makes about these same methods). So this phase generates each
method's CFs **once**, extracts each instance's derived ``t0`` via
``causaltemp_xai.metrics.pns.extract_intervention`` (graph/mechanism-free --
a pure ``x`` vs ``x_cf`` comparison), and reports validity as a function of
the resulting ``T - t0`` -- measured, not controlled, exactly matching the
DoD's own wording ("validity measured as a function of ... T - t0").

**Recourse direction.** Real multi-class data has no single "target class"
convention the way this benchmark's binary synthetic configs do. Fixed to
``target_class=0`` (the first class alphabetically) for every non-target
instance, for all methods -- documented here rather than tuned away, so the
choice is visible to a reader of the result.

Usage
-----
    uv run python experiments/06b_horizon_sweep_real.py --name basicmotions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.methods.counterfactual.cfts_methods import (  # noqa: E402
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
    _DatasetAdapter,
)
from causaltemp_xai.metrics.axis_c import (  # noqa: E402
    ood_plausibility,
    proximity,
    sparsity,
    validity,
)
from causaltemp_xai.metrics.pns import extract_intervention  # noqa: E402
from causaltemp_xai.real_data import DEFAULT_REAL_DIR, load_real_dataset  # noqa: E402
from experiments._common import dump_json, set_run_context, write_csv  # noqa: E402

TARGET_CLASS = 0

#: Horizon bin edges as fractions of T (matches Phase 06's geometric spacing
#: rationale -- decay is expected to concentrate near small T-t0), used to
#: aggregate the per-instance (derived) horizons into a plottable curve.
HORIZON_BIN_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def select_flip_candidates(
    clf, X: np.ndarray, target_class: int, n_cf: int | None = None
) -> np.ndarray:
    preds = clf.predict(X)
    src = [i for i in range(len(X)) if preds[i] != target_class]
    if not src:
        raise SystemExit(f"no instances outside target_class={target_class} to seek CFs for")
    if n_cf is not None:
        src = src[:n_cf]
    return np.asarray(src, dtype=int)


def train_classifier(
    X_train, Y_train, X_val, Y_val, n_classes: int, seed: int = 0
) -> LSTMClassifier:
    clf = LSTMClassifier(
        n_inputs=X_train.shape[-1],
        n_classes=n_classes,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        batch_size=64,
        max_epochs=100,
        patience=10,
        seed=seed,
    )
    clf.fit(X_train, Y_train, X_val, Y_val)
    return clf


def build_methods(X_train, Y_train, target_class: int) -> dict:
    ds = _DatasetAdapter(X_train, Y_train)
    return {
        "CftsWachter": CftsWachterCF(target_class=target_class, dataset=ds, max_cfs=500),
        "CftsCOMTE": CftsCOMTECF(target_class=target_class, dataset=ds),
        "CftsConfeti": CftsConfetiCF(target_class=target_class, dataset=ds),
        "CftsCounts": CftsCountsCF(target_class=target_class, dataset=ds),
        "CftsCels": CftsCelsCF(target_class=target_class, dataset=ds),
    }


def run(
    name: str = "basicmotions",
    out_dir=DEFAULT_REAL_DIR,
    n_cf: int | None = None,
    target_class: int = TARGET_CLASS,
    seed: int = 0,
) -> None:
    set_run_context(seed=seed, config=f"real_{name}")
    data = load_real_dataset(name, out_dir=out_dir)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    n_classes = len(data["meta"]["class_names"])
    T = X_train.shape[1]

    clf = train_classifier(X_train, Y_train, X_val, Y_val, n_classes, seed=seed)
    test_acc = float((clf.predict(X_test) == Y_test).mean())
    print(f"[06b] {name}: T={T} n_classes={n_classes} test_acc={test_acc:.3f}")

    sel_idx = select_flip_candidates(clf, X_test, target_class, n_cf)
    X_sel = X_test[sel_idx]
    print(f"[06b] {len(X_sel)} flip candidates (target_class={target_class})")

    # M4g (`DECISIONS.md` 2026-08-06): persist X_sel + per-method CF arrays,
    # mirroring Phase 03's on-disk contract exactly (`cf/X_sel.npy`,
    # `cf/X_cf_<Method>.npy`). Nothing wrote these before -- only aggregate
    # summary.json/per_instance.csv existed -- so there was nothing for a
    # discovered-graph CF-faith scorer to read. This does not change anything
    # about M4b's own "no causal graph" scope; it only makes the raw CFs
    # available for M4g's separate scoring pass.
    cf_dir = ROOT / "results" / f"real_{name}" / "lstm" / "cf"
    cf_dir.mkdir(parents=True, exist_ok=True)
    np.save(cf_dir / "X_sel.npy", X_sel)

    bin_edges = [max(1, round(f * T)) for f in HORIZON_BIN_FRACTIONS]

    methods = build_methods(X_train, Y_train, target_class)
    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    hdr = f"{'method':<14}{'n':>5}{'no_op':>7}{'mean_h':>8}{'valid':>8}{'prox_l1':>10}{'ood':>8}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for method_name, method in methods.items():
        cfs = np.asarray(method.generate_batch(X_sel, clf), dtype=np.float32)
        np.save(cf_dir / f"X_cf_{method_name}.npy", cfs)
        ood_scores = ood_plausibility(X_train, cfs)  # one IsolationForest fit, whole batch

        n_no_op = 0
        horizons: list[int] = []
        for i in range(len(X_sel)):
            x, cf = X_sel[i], cfs[i]
            t0, nodes, _values = extract_intervention(x, cf)
            no_op = nodes.size == 0
            horizon = None if no_op else int(T - t0)
            if no_op:
                n_no_op += 1
            v = float(validity(cf[None], clf, target_class))
            row = {
                "method": method_name,
                "instance": int(sel_idx[i]),
                "t0": None if no_op else int(t0),
                "horizon": horizon,
                "no_op": no_op,
                "n_intervened_channels": int(nodes.size),
                "validity": v,
                "proximity_l1": float(proximity(x, cf, norm="l1")),
                "proximity_l2": float(proximity(x, cf, norm="l2")),
                "sparsity": float(sparsity(x, cf)),
                "ood": float(ood_scores[i]),
            }
            all_rows.append(row)
            if not no_op:
                horizons.append(horizon)

        scorable = [r for r in all_rows if r["method"] == method_name and not r["no_op"]]
        mean_h = float(np.mean(horizons)) if horizons else float("nan")
        mean_valid = float(np.mean([r["validity"] for r in scorable])) if scorable else float("nan")
        mean_prox = (
            float(np.mean([r["proximity_l1"] for r in scorable])) if scorable else float("nan")
        )
        mean_ood = float(np.mean([r["ood"] for r in scorable])) if scorable else float("nan")
        print(
            f"{method_name:<14}{len(X_sel):>5}{n_no_op:>7}{mean_h:>8.1f}"
            f"{mean_valid:>8.2f}{mean_prox:>10.2f}{mean_ood:>8.2f}"
        )

        # Bin scorable instances by derived horizon for a plottable curve.
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = [
                r
                for r in scorable
                if lo <= r["horizon"] < hi or (hi == bin_edges[-1] and r["horizon"] == hi)
            ]
            if not in_bin:
                continue
            summary_rows.append(
                {
                    "method": method_name,
                    "horizon_bin_lo": lo,
                    "horizon_bin_hi": hi,
                    "n": len(in_bin),
                    "validity_mean": float(np.mean([r["validity"] for r in in_bin])),
                    "proximity_l1_mean": float(np.mean([r["proximity_l1"] for r in in_bin])),
                    "sparsity_mean": float(np.mean([r["sparsity"] for r in in_bin])),
                    "ood_mean": float(np.mean([r["ood"] for r in in_bin])),
                }
            )

    out_root = ROOT / "results" / f"real_{name}"
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "per_instance.csv", all_rows)
    dump_json(
        out_root / "summary.json",
        {
            "seed": seed,
            "dataset": name,
            "T": T,
            "n_classes": n_classes,
            "target_class": target_class,
            "test_acc": test_acc,
            "graph": None,
            "note": (
                "M4b: no causal graph, CF-faith/PNS/do-complexity not computed "
                "(no oracle mechanism on real data). Horizon is DERIVED per "
                "instance via extract_intervention, not swept -- these 5 "
                "methods have no t0 parameter to control."
            ),
            "horizon_bins": summary_rows,
        },
    )
    print(f"\n[06b] wrote {out_root / 'per_instance.csv'}")
    print(f"[06b] wrote {out_root / 'summary.json'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="basicmotions")
    parser.add_argument("--out-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=TARGET_CLASS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run(args.name, args.out_dir, args.n_cf, args.target_class, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
