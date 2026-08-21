"""Phase 03: Run CF explainers on a trained classifier.

Loads the dataset + LSTM checkpoint for one config, selects the flip
candidates (test instances not already predicted as ``TARGET_CLASS``), runs
every registered CF method (CARLA + PearlCARLA + cfts-backed Wachter/COMTE/
CONFETTI/CounTS/CELS + TSCausalCF), and persists the raw counterfactual arrays. Also
computes every CF method's Shift-VR-lite robustness metric (Axis B), which
needs live method/model access and so belongs here rather than in the
metrics-only Phase 04.

**Attribution and Axis-A were removed from this phase 2026-08-03**
 along with ``causaltemp_xai/methods/attribution/``. Those
method families were descoped 2026-07-29 and no axis here scores them, but the
code stayed wired in, so every run paid for work no contribution claims.

Every axis that scores counterfactual explanations (Axis C + CF-faith here in
Phase 04/05, Axis B's Shift-VR here in Phase 03) is run over the **full** set
of selected CF methods -- no method is singled out or excluded from an
applicable axis. Wachter is the cfts-backed gradient implementation
(``CftsWachterCF``, wrapping the genuine vendored ``cfts`` library); the
native from-scratch ``WachterCF`` reimplementation was removed 2026-08-05
 as redundant with it.

Nonlinear configs (``smoke_nl``/``full_nl``) are supported here too, given a
trained classifier (Phase 02 now trains on any config -- see its docstring).
This goes **beyond** the locked NlinearSCM-T plan's scope: real CF methods on
the nonlinear mechanism were deferred to a collaborator's track and were never
validated there (historical: ``docs/archive/plans/nlinearscm-t/``). It works
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
    cf/no_cf_found_<Method>.npy    Boolean failed-search status, (n_cf,)
    cf/no_cf_found_provenance.json generated vs inferred status source per method
    cf/recourse_diagnostics_*.json CARLA/PearlCARLA lambda-backoff diagnostics
    shift_vr.json                 Axis B: validity-retention under a noise shift, all CF methods

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
    TSCausalCF,
)
from causaltemp_xai.methods.counterfactual.cfts_methods import _DatasetAdapter  # noqa: E402
from experiments._common import (  # noqa: E402
    TARGET_CLASS,
    config_dir,
    dump_json,
    select_flip_candidates,
    set_run_context,
)


def build_methods(X_train, y_train, target_class: int = TARGET_CLASS) -> dict:
    """The full set of selected CF methods -- every axis below runs on all of
    them. ``CftsWachter`` is the cfts-backed Wachter implementation; the
    native from-scratch ``WachterCF`` was removed 2026-08-05 as a redundant
    duplicate.

    ``target_class`` defaults to the pipeline-wide ``TARGET_CLASS`` and is
    parameterised only so Phase 07's necessity direction can build the same
    method set aimed the *other* way (target -> non-target). Nothing in phases
    01-05 passes it; the main pipeline's behaviour is unchanged.

    """
    ds = _DatasetAdapter(X_train, y_train)
    tc = target_class
    return {
        "CARLA": CARLARecourse(target_class=tc, n_steps=300, t0_fractions=(0.25, 0.5)),
        "PearlCARLA": PearlCARLARecourse(target_class=tc, t0_fractions=(0.25, 0.5)),
        "CftsWachter": CftsWachterCF(target_class=tc, dataset=ds, max_cfs=500),
        "CftsCOMTE": CftsCOMTECF(target_class=tc, dataset=ds),
        "CftsConfeti": CftsConfetiCF(target_class=tc, dataset=ds),
        "CftsCounts": CftsCountsCF(target_class=tc, dataset=ds),
        "CftsCels": CftsCelsCF(target_class=tc, dataset=ds),
        "TSCausal": TSCausalCF(target_class=tc),
    }


def _call_batch(batch_fn, X, clf, graph, mech):
    """Call a bound batch method with its declared causal arguments."""
    params = inspect.signature(batch_fn).parameters
    if "graph" in params or "mechanism" in params:
        return batch_fn(X, clf, graph, mech)
    return batch_fn(X, clf)


def generate_cfs(method, X, clf, graph, mech, *, with_status: bool = False):
    """Generate one CF per instance, optionally returning failed-search status.

    The default remains the historical ndarray-only return used by Phase 07.
    Phase 03 requests status explicitly. Methods without a status API receive
    an all-false vector whose provenance is marked ``inferred``.
    """
    status_fn = getattr(method, "generate_batch_with_status", None)
    if with_status and callable(status_fn):
        cfs, no_cf_found = _call_batch(status_fn, X, clf, graph, mech)
        status_source = "generated"
    else:
        cfs = _call_batch(method.generate_batch, X, clf, graph, mech)
        no_cf_found = np.zeros(len(X), dtype=bool)
        status_source = "inferred"

    cfs = np.asarray(cfs, dtype=np.float32)
    if not with_status:
        return cfs

    no_cf_found = np.asarray(no_cf_found)
    if no_cf_found.dtype != np.bool_:
        raise TypeError(f"no_cf_found must have bool dtype, got {no_cf_found.dtype}")
    if no_cf_found.shape != (len(cfs),):
        raise ValueError(
            f"no_cf_found shape {no_cf_found.shape} does not match CF batch ({len(cfs)},)"
        )
    status_info = {
        "source": status_source,
        "n": len(cfs),
        "n_no_cf_found": int(no_cf_found.sum()),
    }
    return cfs, no_cf_found, status_info


def load_or_make_shift_test(cfg, out_dir):
    shift_cfg = shifted_config(cfg, noise_type="uniform")
    dest = Path(out_dir) / shift_cfg.name
    if not (dest / "meta.json").exists():
        generate_and_save(shift_cfg, out_dir=out_dir)
    data = load_dataset(shift_cfg.name, out_dir=out_dir)
    return data["X_test"]


def run(
    config_name: str,
    n_cf: int,
    out_dir,
    methods_filter=None,
    seed: int | None = None,
    skip_aux: bool = False,
) -> None:
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
    status_provenance: dict[str, dict] = {}
    for name, method in all_methods.items():
        print(f"[03] generating CFs: {name} ...")
        try:
            cfs, no_cf_found, status_info = generate_cfs(
                method, X_sel, clf, graph, mech, with_status=True
            )
            np.save(cf_dir / f"X_cf_{name}.npy", cfs)
            np.save(cf_dir / f"no_cf_found_{name}.npy", no_cf_found)
            generated[name] = cfs
            status_provenance[name] = status_info
            diagnostics = getattr(method, "last_batch_diagnostics", None)
            if diagnostics is not None:
                dump_json(
                    cf_dir / f"recourse_diagnostics_{name}.json",
                    {"method": name, "instances": diagnostics},
                )
            print(
                f"     -> {cf_dir / f'X_cf_{name}.npy'}; "
                f"no_cf_found={int(no_cf_found.sum())}/{len(no_cf_found)} "
                f"({status_info['source']})"
            )
        except Exception as exc:
            print(f"     {name} FAILED: {exc}")

    dump_json(cf_dir / "no_cf_found_provenance.json", {"methods": status_provenance})

    if skip_aux:
        # CF arrays are written and complete; the auxiliary block below
        # (Shift-VR) does not depend on which --methods ran
        # and cost ~8 min per invocation. Skipping them lets CF generation be
        # run in small chunks -- the only way to make progress on a host where
        # long processes are being killed sporadically, since X_cf_*.npy files
        # accumulate across invocations. Run once WITHOUT --skip-aux (or with
        # the full roster) to produce the auxiliary artifacts.
        print("[03] --skip-aux: Shift-VR NOT computed.")
        print("[03] done (CF arrays only).")
        return

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
    parser = argparse.ArgumentParser(description="Run CF methods + shift-VR.")
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--n-cf", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of method names to run.")
    parser.add_argument(
        "--skip-aux",
        action="store_true",
        help="Write CF arrays only; skip Shift-VR "
        "(~8 min of per-invocation overhead). For chunked CF generation.",
    )
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
    run(
        args.config,
        n_cf,
        args.out_dir,
        methods_filter=args.methods,
        seed=args.seed,
        skip_aux=args.skip_aux,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
