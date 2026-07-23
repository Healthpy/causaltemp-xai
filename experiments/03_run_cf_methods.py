"""Phase 03: Run CF explainers + the IG attribution foil on a trained classifier.

Loads the dataset + LSTM checkpoint for one config, selects the flip
candidates (test instances not already predicted as ``TARGET_CLASS``), runs
every registered CF method (CARLA + PearlCARLA + cfts-backed Wachter/COMTE/
CONFETTI/CounTS/CELS), and persists the raw counterfactual arrays. Also
evaluates the Integrated-Gradients attribution foil on its **suitable** axes -- Axis A
(causal-relevance of the saliency map, scored against ground-truth oracle
interventions) and Axis D (input-sensitivity) -- plus every CF method's
Shift-VR-lite robustness metric (Axis D). All of these need live
method/model access, so they belong here rather than in the metrics-only
Phase 04.

Every axis that scores counterfactual explanations (Axis C + CF-faith here in
Phase 04/05, Axis D's Shift-VR here in Phase 03) is run over the **full** set
of selected CF methods -- no method is singled out or excluded from an
applicable axis. Wachter is the cfts-backed gradient implementation
(``CftsWachterCF``); the native from-scratch ``WachterCF`` is not used here.

Nonlinear configs (``smoke_nl``/``full_nl``) are supported here too, given a
trained classifier (Phase 02 now trains on any config -- see its docstring).
This goes **beyond** the locked NlinearSCM-T plan's scope: real CF methods on
the nonlinear mechanism were deferred to a collaborator's track and were never
validated there (``docs/plans/nlinearscm-t/index.md`` Backlog #2). It works
mechanically because every method here is mechanism-generic -- the cfts-*
methods never touch the SCM at all, and CARLA/Axis-A's oracle interventions
route through ``mechanism.forward_torch``/``forward_numpy``, which
``MLPMechanism`` implements just like ``LinearMechanism`` -- but CARLA's
recourse objective and the cfts baselines were only ever tuned/validated
against the linear VAR mechanism, so treat nonlinear results here as
exploratory, not a validated benchmark claim.

Outputs (under ``results/<config>/lstm/``)::

    cf/X_sel.npy                  selected factual instances, (n_cf, T, k)
    cf/X_cf_<Method>.npy           one array per CF method, (n_cf, T, k)
    attribution.json              IG deletion/insertion-AUC summary
    axis_a_attribution.json       Axis A: ICC / causal-coverage of the IG saliency map
    shift_vr.json                 Axis D: validity-retention under a noise shift, all CF methods

Usage
-----
    uv run python experiments/03_run_cf_methods.py --config smoke --n-cf 20
    uv run python experiments/03_run_cf_methods.py --config full --n-cf 100 --methods CftsWachter CARLA
    uv run python experiments/03_run_cf_methods.py --config smoke_nl --n-cf 10   # exploratory
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config, seeded_variant, shifted_config  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, generate_and_save, load_dataset  # noqa: E402
from causaltemp_xai.eval import shift_vr  # noqa: E402
from causaltemp_xai.methods import (  # noqa: E402  # noqa: E402
    CARLARecourse,
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsWachterCF,
    PearlCARLARecourse,
)
from causaltemp_xai.methods.attribution import (  # noqa: E402
    deletion_curve,
    insertion_curve,
    integrated_gradients,
)
from causaltemp_xai.methods.counterfactual.cfts_methods import _DatasetAdapter  # noqa: E402
from causaltemp_xai.metrics.axis_d import input_sensitivity  # noqa: E402
from experiments._common import (  # noqa: E402
    axis_a_for_attribution,
    build_oracle_interventions,
    causal_parents,
    config_dir,
    dump_json,
    select_flip_candidates,
    set_run_context,
)

TARGET_CLASS = 1


def build_methods(X_train, y_train) -> dict:
    """The full set of selected CF methods -- every axis below runs on all of
    them. ``CftsWachter`` (cfts-backed) replaces the native ``WachterCF``.

    ``n_steps`` decision for ``PearlCARLA`` (M2 gap-closure, see
    ``docs/m2_multiseed_and_pearl_carla.md`` S3 "Wiring PearlCARLA into the
    phase-03 registry" for the full rationale): ``CARLA`` below overrides
    ``n_steps=300`` (its class default is 500) purely for pipeline speed.
    ``PearlCARLA`` deliberately does **not** override ``n_steps`` here, so it
    runs at its own class default of 500 -- matching exactly the step count
    its ``lam_prox=0.1`` default was empirically validated at (doc S2.3:
    "full n_steps=500 ... not the reduced value used only for the diagnostic
    hyperparameter sweep"). Silently mirroring CARLA's 300 would move
    PearlCARLA off its one validated operating point with no new evidence
    that its default still behaves the same way there. The resulting
    asymmetry (CARLA@300 vs PearlCARLA@500) is intentional, not an oversight,
    and costs little at smoke scale (k=5, T=30) -- it would need revisiting
    before any full-scale run given full_nl's much longer horizon (see the
    same doc section's note on `lam_prox` needing to shrink further there).
    """
    ds = _DatasetAdapter(X_train, y_train)
    return {
        "CARLA": CARLARecourse(target_class=TARGET_CLASS, n_steps=300, t0_fractions=(0.25, 0.5)),
        "PearlCARLA": PearlCARLARecourse(target_class=TARGET_CLASS, t0_fractions=(0.25, 0.5)),
        "CftsWachter": CftsWachterCF(target_class=TARGET_CLASS, dataset=ds, max_cfs=500),
        "CftsCOMTE": CftsCOMTECF(target_class=TARGET_CLASS, dataset=ds),
        "CftsConfeti": CftsConfetiCF(target_class=TARGET_CLASS, dataset=ds),
        "CftsCounts": CftsCountsCF(target_class=TARGET_CLASS, dataset=ds),
        "CftsCels": CftsCelsCF(target_class=TARGET_CLASS, dataset=ds),
    }


def generate_cfs(method, X, clf, graph, mech) -> np.ndarray:
    """Generate one CF per instance, routing graph/mechanism to causal methods."""
    params = inspect.signature(method.generate_batch).parameters
    if "graph" in params or "mechanism" in params:
        cfs = method.generate_batch(X, clf, graph, mech)
    else:
        cfs = method.generate_batch(X, clf)
    return np.asarray(cfs, dtype=np.float32)


def attribution_block(clf, X_sel, ig_steps=64, curve_steps=50):
    """IG deletion/insertion-AUC foil, plus the raw ``(N, T, k)`` saliency maps
    (needed by :func:`axis_a_block` / Axis D input-sensitivity)."""
    del_aucs, ins_aucs, maps = [], [], []
    for x in X_sel:
        ig = integrated_gradients(clf, x, TARGET_CLASS, steps=ig_steps)
        _, del_auc = deletion_curve(clf, x, ig, TARGET_CLASS, n_steps=curve_steps)
        _, ins_auc = insertion_curve(clf, x, ig, TARGET_CLASS, n_steps=curve_steps)
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
        maps.append(np.asarray(ig, dtype=float))
    summary = {
        "method": "IntegratedGradients",
        "deletion_auc": float(np.mean(del_aucs)),
        "insertion_auc": float(np.mean(ins_aucs)),
        "insertion_minus_deletion": float(np.mean(ins_aucs) - np.mean(del_aucs)),
        "n": len(X_sel),
    }
    return summary, np.stack(maps)


def axis_a_block(clf, X_sel, attributions, graph, mech, ig_steps=64) -> dict:
    """Axis A (ICC + causal-coverage) for the IG saliency map, plus Axis D
    input-sensitivity -- both scored against ground-truth oracle interventions
    so ``int_channel`` / ``causal_parents`` are real, not proxies."""
    interventions = build_oracle_interventions(X_sel, mech)
    int_channels = [node for (_, node, _) in interventions]
    t0s = [t0 for (t0, _, _) in interventions]
    causal_parents_list = [causal_parents(graph, node) for node in int_channels]

    axis_a = axis_a_for_attribution(attributions, int_channels, causal_parents_list, t0s=t0s)

    def _attribution_fn(x):
        return integrated_gradients(clf, x, TARGET_CLASS, steps=ig_steps)

    axis_a["InputSens"] = input_sensitivity(np.asarray(X_sel), _attribution_fn, n_trials=5)
    axis_a["n"] = len(X_sel)
    return axis_a


def load_or_make_shift_test(cfg, out_dir):
    shift_cfg = shifted_config(cfg, noise_type="uniform")
    dest = Path(out_dir) / shift_cfg.name
    if not (dest / "meta.json").exists():
        generate_and_save(shift_cfg, out_dir=out_dir)
    data = load_dataset(shift_cfg.name, out_dir=out_dir)
    return data["X_test"]


def run(config_name: str, n_cf: int, out_dir, methods_filter=None, seed: int | None = None) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    set_run_context(seed=cfg.seed, config=cfg.name)
    if cfg.mechanism_type != "linear":
        print(
            f"[03] NOTE: '{config_name}' is a nonlinear (MLP) config -- CF methods here are "
            "exploratory, not validated by the locked plan (Backlog #2). See module docstring."
        )

    data = load_dataset(cfg.name, out_dir=out_dir)
    X_train, y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    graph, mech = data["graph"], data["mechanism"]

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"no checkpoint at {ckpt}; train first: "
            f"uv run python experiments/02_train_classifiers.py --config {cfg.name}"
        )
    clf = LSTMClassifier.load(ckpt)
    test_acc = clf.score(X_test, Y_test)

    sel = select_flip_candidates(clf, X_test, n_cf)
    X_sel = X_test[sel]
    print(
        f"[03] config={cfg.name} k={cfg.k} T={cfg.T} | test_acc={test_acc:.3f} | "
        f"{len(X_sel)} flip candidates"
    )

    all_methods = build_methods(X_train, y_train)
    if methods_filter:
        all_methods = {k: v for k, v in all_methods.items() if k in methods_filter}

    out = config_dir(cfg.name, "lstm")
    cf_dir = out / "cf"
    cf_dir.mkdir(parents=True, exist_ok=True)
    np.save(cf_dir / "X_sel.npy", X_sel)

    # Keep the generated arrays: Shift-VR's base half is exactly this work, so
    # handing them over below saves a full redundant generation pass.
    generated: dict[str, np.ndarray] = {}
    for name, method in all_methods.items():
        print(f"[03] generating CFs: {name} ...")
        try:
            cfs = generate_cfs(method, X_sel, clf, graph, mech)
            np.save(cf_dir / f"X_cf_{name}.npy", cfs)
            generated[name] = cfs
            print(f"     -> {cf_dir / f'X_cf_{name}.npy'}")
        except Exception as exc:
            print(f"     {name} FAILED: {exc}")

    print("[03] integrated-gradients attribution foil ...")
    attribution, attr_maps = attribution_block(clf, X_sel)
    dump_json(out / "attribution.json", attribution)
    print(f"     -> {out / 'attribution.json'}")

    print("[03] axis A (causal-relevance of IG saliency) + axis D (input-sensitivity) ...")
    axis_a = axis_a_block(clf, X_sel, attr_maps, graph, mech)
    dump_json(out / "axis_a_attribution.json", axis_a)
    print(
        f"     ICC={axis_a['ICC']:.3f} MCC_coverage={axis_a['MCC_coverage']:.3f} "
        f"InputSens={axis_a['InputSens']:.3f}"
    )
    print(f"     -> {out / 'axis_a_attribution.json'}")

    # Only methods that actually produced base CFs above: their arrays are
    # reused as Shift-VR's base half (no regeneration), and a method that
    # failed here cannot yield a base validity anyway -- previously it was
    # passed on regardless and raised *uncaught* inside shift_vr, killing the
    # phase after all the expensive work was already done.
    shift_methods = {n: m for n, m in all_methods.items() if n in generated}
    skipped = [n for n in all_methods if n not in generated]
    print(f"[03] shift-VR-lite ({len(shift_methods)} CF methods; base CFs reused) ...")
    if skipped:
        print(f"     skipping (no base CFs generated): {', '.join(skipped)}")
    X_shift_test = load_or_make_shift_test(cfg, out_dir)
    shift_sel = select_flip_candidates(clf, X_shift_test, n_cf)
    X_shift_sel = X_shift_test[shift_sel]
    shift = shift_vr(
        clf,
        shift_methods,
        X_sel,
        X_shift_sel,
        graph,
        mech,
        TARGET_CLASS,
        cf_base=generated,
    )
    dump_json(out / "shift_vr.json", shift)
    print(f"     -> {out / 'shift_vr.json'}")

    print("[03] done.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run CF methods + attribution + shift-VR.")
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of method names to run.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Multi-seed replicate (M2): override --config's registered seed "
            "via causaltemp_xai.config.seeded_variant, reading the seeded "
            "dataset/checkpoint and writing under '<config>_seed<seed>'."
        ),
    )
    args = parser.parse_args(argv)

    n_cf = args.n_cf or (20 if args.config.startswith("smoke") else 100)
    run(args.config, n_cf, args.out_dir, methods_filter=args.methods, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
