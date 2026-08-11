"""Phase 07b: discovered-graph CF-faith on Tier 2 real data, uncertainty-
quantified (M4g, `DECISIONS.md` 2026-08-06).

**Reopens part of the 2026-07-31 M4b decision** that discovered-graph
CF-faith scoring on real data "stays descoped." M4f (2026-08-06) built a
bootstrap ensemble for DYNOTEARS on synthetic data (Tier 1): B independent
refits on resamples of a dataset's trajectories, reporting the *spread*
across the ensemble rather than trusting one arbitrary fit. This phase
applies the same idea to Tier 2, where there is no true mechanism to mask
(unlike Tier 1's `build_masked_mechanism`) -- instead, each ensemble
member's own fitted DYNOTEARS weights are used directly as an approximate
linear mechanism (`DYNOTEARS.to_linear_mechanism`,
`causaltemp_xai/methods/causal/dynotears.py`).

**What this resolves and what it does not, stated plainly (not discovered
later).** The ensemble quantifies *estimation variance* -- how much the
inferred graph/weights wobble under resampling of the same-size sample from
the same process. It does **not** quantify *model-misspecification bias*:
every ensemble member shares DYNOTEARS's linear structural-equation
assumption, so if that assumption is wrong for this (real, plausibly
nonlinear -- e.g. motion-capture) data, all B members can agree tightly
while being systematically wrong together. This is `docs/risk_register.md`
RISK-22, and every result this phase writes carries that limitation
alongside it, not as a footnote to be discovered separately.

**Prerequisite.** `experiments/06b_horizon_sweep_real.py` must have already
run for the target dataset -- it persists `X_sel.npy`/`X_cf_<Method>.npy`
under `results/real_<name>/lstm/cf/`, which this phase reads and does not
regenerate.

**Ensemble fit sample.** Uses `X_train + X_val` combined (`BasicMotions`:
N=32+8=40) for the DYNOTEARS refits specifically -- the train-only split
(N=32) is small for bootstrap resampling. The CF-generation split in
`06b_horizon_sweep_real.py` is untouched.

**Lag order.** Fixed at `L=1` -- a stated assumption (real data's true lag
structure is unknown), not a computed fact, disclosed here rather than
silently chosen.

Usage
-----
    uv run python experiments/07b_discovered_graph_real.py --name basicmotions --ensemble-b 5
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.methods.causal import DYNOTEARS, PCMCIPlus  # noqa: E402
from causaltemp_xai.metrics.axis_a import shd  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402
from causaltemp_xai.real_data import DEFAULT_REAL_DIR, load_real_dataset  # noqa: E402
from causaltemp_xai.stats import bootstrap_ci, bootstrap_resample_indices  # noqa: E402
from experiments._common import dump_json, set_run_context  # noqa: E402

# Phase 07 is a numbered-prefix module (not a valid `import` target), loaded
# the way tests/test_experiment_registry.py and tests/test_auxiliary_methods.py
# already do, to reuse _mean_soft_cf_faith rather than duplicate it.
_phase07 = importlib.import_module("experiments.07_auxiliary_methods")
_mean_soft_cf_faith = _phase07._mean_soft_cf_faith

LIMITATION_NOTE = (
    "cf_faith_discovered_* quantifies estimation VARIANCE (resampling stability of the "
    "DYNOTEARS fit across B independent refits), not model-misspecification BIAS. Every "
    "ensemble member shares DYNOTEARS's linear structural-equation assumption; if that "
    "assumption is wrong for this real (plausibly nonlinear) data, all B members can agree "
    "tightly while being systematically wrong together. A tight ensemble does not rule this "
    "out. See docs/risk_register.md RISK-22 and ROADMAP.md M4g."
)


def score_method_discovered(X_sel, cfs, mechanisms, seed=0) -> dict:
    """The ``cf_faith_discovered_*`` row for one method's persisted CFs,
    scored against every ensemble member's mechanism and aggregated via
    :func:`bootstrap_ci`.

    Extracted as a standalone, directly-testable function (not left inline
    in :func:`run`'s loop) so it has a single real producer to check the
    taxonomy's declared Axis-C keys against, matching this repo's own
    standing discipline (`tests/test_taxonomy.py`::
    ``test_the_experiments_layer_emits_what_it_is_credited_with`` -- "a
    taxonomy that lists metrics the pipeline does not produce is as wrong as
    one that misses metrics it does").

    Parameters
    ----------
    X_sel, cfs:
        Factual and counterfactual arrays, ``(n, T, k)``.
    mechanisms:
        The ``B`` ensemble members' :class:`LinearMechanism` objects (from
        :func:`_fit_mechanism_ensemble`).
    seed:
        Resampling seed for the two :func:`bootstrap_ci` calls.
    """
    k = X_sel.shape[-1]
    L = mechanisms[0].L if mechanisms else 1
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    # CFfaith.score's `graph` parameter is not used directly in computation
    # (kept for API completeness / downstream analysis) -- there is no true
    # graph for Tier 2, so an all-ones placeholder of the right shape is
    # passed through, never read for correctness.
    graph_placeholder = np.ones((k, k, L))

    rollout_scores = [
        _mean_soft_cf_faith(X_sel, cfs, graph_placeholder, mech, rollout) for mech in mechanisms
    ]
    pearl_scores = [
        _mean_soft_cf_faith(X_sel, cfs, graph_placeholder, mech, pearl) for mech in mechanisms
    ]
    r_ci = bootstrap_ci(rollout_scores, seed=seed)
    p_ci = bootstrap_ci(pearl_scores, seed=seed)
    return {
        "cf_faith_discovered_rollout_mean": r_ci.mean,
        "cf_faith_discovered_rollout_ci_lo": r_ci.ci_lo,
        "cf_faith_discovered_rollout_ci_hi": r_ci.ci_hi,
        "cf_faith_discovered_pearl_mean": p_ci.mean,
        "cf_faith_discovered_pearl_ci_lo": p_ci.ci_lo,
        "cf_faith_discovered_pearl_ci_hi": p_ci.ci_hi,
        "ensemble_b": len(mechanisms),
    }


def _fit_mechanism_ensemble(X_all, k, L, B, seed=0, threshold=0.1):
    """B independent DYNOTEARS refits on bootstrap resamples of ``X_all``'s
    N-axis, each converted to a :class:`LinearMechanism` via
    :meth:`DYNOTEARS.to_linear_mechanism`.

    Deliberately **not** a reuse of `07_auxiliary_methods._fit_dynotears_ensemble`
    (M4f): that function returns only density-matched adjacency arrays, which
    is what Tier 1's `graph_error` ensemble needs. Tier 2 needs the fitted
    model itself (to call `to_linear_mechanism`), so this is a parallel,
    small loop over the same resampling primitive
    (`causaltemp_xai.stats.bootstrap_resample_indices`), not a modification
    of the closed, tested M4f function.
    """
    idx = bootstrap_resample_indices(n=X_all.shape[0], n_boot=B, seed=seed)
    mechanisms = []
    for b in range(B):
        model = DYNOTEARS(k=k, p=L).fit(X_all[idx[b]])
        mechanisms.append(model.to_linear_mechanism(threshold=threshold))
    return mechanisms


def run(
    name: str = "basicmotions",
    out_dir=DEFAULT_REAL_DIR,
    ensemble_b: int = 5,
    seed: int = 0,
    threshold: float = 0.1,
) -> None:
    set_run_context(seed=seed, config=f"real_{name}")
    data = load_real_dataset(name, out_dir=out_dir)
    X_train, X_val = data["X_train"], data["X_val"]
    X_all = np.concatenate([X_train, X_val], axis=0)
    k = X_all.shape[-1]
    L = 1  # stated assumption -- see module docstring

    cf_dir = ROOT / "results" / f"real_{name}" / "lstm" / "cf"
    x_sel_path = cf_dir / "X_sel.npy"
    if not x_sel_path.exists():
        raise SystemExit(
            f"[07b] no persisted CF arrays at {cf_dir}; run "
            f"`uv run python experiments/06b_horizon_sweep_real.py --name {name}` first "
            "(M4g needs the X_sel.npy/X_cf_*.npy it now writes)."
        )
    X_sel = np.load(x_sel_path)

    print(
        f"[07b] fitting DYNOTEARS ensemble (B={ensemble_b}) on "
        f"{X_all.shape[0]} sequences (k={k}, p={L}, train+val combined) ..."
    )
    mechanisms = _fit_mechanism_ensemble(X_all, k, L, ensemble_b, seed=seed, threshold=threshold)

    method_rows = []
    hdr = f"{'method':<14}{'rollout_mean':>14}{'rollout_ci':>18}{'pearl_mean':>12}{'pearl_ci':>18}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for cf_path in sorted(cf_dir.glob("X_cf_*.npy")):
        method = cf_path.stem[len("X_cf_") :]
        cfs = np.load(cf_path)

        row = score_method_discovered(X_sel, cfs, mechanisms, seed=seed)
        method_rows.append({"method": method, **row})
        r_lo, r_hi = (
            row["cf_faith_discovered_rollout_ci_lo"],
            row["cf_faith_discovered_rollout_ci_hi"],
        )
        p_lo, p_hi = row["cf_faith_discovered_pearl_ci_lo"], row["cf_faith_discovered_pearl_ci_hi"]
        r_ci_str = f"[{r_lo:.3f},{r_hi:.3f}]"
        p_ci_str = f"[{p_lo:.3f},{p_hi:.3f}]"
        print(
            f"{method:<14}{row['cf_faith_discovered_rollout_mean']:>14.3f}{r_ci_str:>18}"
            f"{row['cf_faith_discovered_pearl_mean']:>12.3f}{p_ci_str:>18}"
        )

    out = {
        "provenance": {
            "dataset": name,
            "ensemble_b": ensemble_b,
            "threshold": threshold,
            "lag_order_L": L,
            "n_ensemble_fit": int(X_all.shape[0]),
        },
        "methods": method_rows,
        "limitation": LIMITATION_NOTE,
    }
    out_dir_res = ROOT / "results" / f"real_{name}" / "dynotears"
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "cf_faith_discovered.json", out)
    print(f"\n[07b] wrote {out_dir_res / 'cf_faith_discovered.json'}")


def run_cross_method_agreement_real(
    name: str,
    out_dir=DEFAULT_REAL_DIR,
    pc_alpha: float = 0.05,
    threshold: float = 0.1,
    seed: int = 0,
) -> None:
    """DYNOTEARS-vs-PCMCIplus cross-method agreement check on real Tier-2 data
    (M4i, `DECISIONS.md` 2026-08-06) -- the real-data counterpart of
    `07_auxiliary_methods.run_cross_method_agreement` (Tier 1, synthetic).

    Real data has **no ground truth graph**, so unlike the Tier-1 version this
    reports `shd_between_methods` only -- structural agreement between the two
    methods' own independently inferred graphs -- and **no AUC** (there is
    nothing to be accurate *against*). This is a materially weaker check than
    Tier 1's: agreement here is not evidence of correctness, only of the two
    structurally different model classes landing on similar structure. Stated
    explicitly in the written output, never conflated with Tier 1's numbers.

    Single fit each (not the M4f/M4g bootstrap ensemble -- that machinery
    exists for `run`'s discovered-*mechanism* CF-faith pipeline above and is a
    separate concern). No density-matching either: `_density_matched_adjacency`
    needs a true edge count, which does not exist here -- each method's own
    `inferred_graph(max_lag, threshold)` thresholding is used directly, the
    same choice `run`'s ensemble path already makes for the same reason.
    """
    set_run_context(seed=seed, config=f"real_{name}")
    data = load_real_dataset(name, out_dir=out_dir)
    X_train, X_val = data["X_train"], data["X_val"]
    X_all = np.concatenate([X_train, X_val], axis=0)
    k = X_all.shape[-1]
    L = 1  # stated assumption -- see module docstring

    print(f"[07b] cross-method agreement: fitting DYNOTEARS on {X_all.shape[0]} sequences ...")
    dyn_model = DYNOTEARS(k=k, p=L).fit(X_all)
    dyn_adj, _ = dyn_model.inferred_graph(max_lag=L, threshold=threshold)

    print(f"[07b] cross-method agreement: fitting PCMCIplus on {X_all.shape[0]} sequences ...")
    pcmci_model = PCMCIPlus(k=k, tau_max=L, pc_alpha=pc_alpha).fit(X_all)
    pcmci_adj, _ = pcmci_model.inferred_graph(max_lag=L, threshold=threshold)

    out = {
        "provenance": {
            "dataset": name,
            "pc_alpha": pc_alpha,
            "threshold": threshold,
            "lag_order_L": L,
            "n_fit": int(X_all.shape[0]),
        },
        "shd_between_methods": float(shd(dyn_adj, pcmci_adj)),
        "note": (
            "structural agreement only, no correctness claim possible without ground truth -- "
            "real data has no true graph to compare either method against. See "
            "docs/risk_register.md RISK-22 and the Tier-1 version of this check "
            "(experiments/07_auxiliary_methods.py::run_cross_method_agreement) for the "
            "AUC-vs-truth comparison this cannot provide."
        ),
    }
    out_dir_res = ROOT / "results" / f"real_{name}" / "cross_method_agreement"
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_agreement.json", out)

    print(
        f"[07b] cross-method agreement: SHD(dynotears,pcmciplus)={out['shd_between_methods']:.0f}"
    )
    print(f"[07b] wrote {out_dir_res / 'graph_agreement.json'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="basicmotions")
    parser.add_argument("--out-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument(
        "--ensemble-b",
        type=int,
        default=5,
        help="Number of independent DYNOTEARS refits (M4f/M4g escalation discipline: "
        "start small, escalate only after timing is measured).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--mode",
        default="discovered_graph",
        choices=("discovered_graph", "cross_method_agreement"),
        help="discovered_graph (default) -- DYNOTEARS ensemble + Tier-2 CF-faith (this "
        "file's original purpose, M4g); cross_method_agreement -- DYNOTEARS-vs-PCMCIplus "
        "graph agreement, no CF-faith (M4i).",
    )
    parser.add_argument("--pc-alpha", type=float, default=0.05, help="cross_method_agreement only")
    args = parser.parse_args(argv)
    if args.mode == "cross_method_agreement":
        run_cross_method_agreement_real(
            args.name, args.out_dir, args.pc_alpha, args.threshold, args.seed
        )
    else:
        run(args.name, args.out_dir, args.ensemble_b, args.seed, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
