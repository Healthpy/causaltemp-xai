"""Phase 08: horizon sweep — validity and CF-faith as functions of ``T - t0`` (M4d).

Contribution #2 is horizon impossibility: a single causal intervention placed
far enough from the label site cannot move the classifier, however faithful it
is. Phases 01-05 evaluate that claim at exactly **two** fixed points
(``t0_fractions = (0.25, 0.5)``), and both fail — which lets the claim be
*asserted* but not *plotted*. This phase sweeps ``t0`` so the decay curve and
the horizon at which validity collapses become measurable.

**What this phase does not do.** It does not change ``t0_fractions``. That
default stays ``(0.25, 0.5)`` and changing it is recorded as *rejected, not
deferred* (`ROADMAP.md` Descoped Items, 2026-07-31): PearlCARLA's
``validity = 0.00`` under the default **is** the result, so moving ``t0`` later
would select the regime where the method succeeds and delete the finding. This
phase is purely additive — it writes to ``results/<config>/horizon/`` and
touches no committed row.

**Why this is not a config variant.** ``t0`` is a *method* parameter
(``CARLARecourse(t0_steps=...)``), not a dataset parameter. The SCM, the
dataset, the split and the classifier are all invariant to it. So this phase
loads them **once** and re-runs only CF generation per horizon — no preset, no
regeneration, no retraining. It also reuses Phase 03's ``X_sel.npy``, so every
sweep point scores the *same instances* the main pipeline reports on and the
curve is directly comparable to the committed tables.

**Which methods are swept, and why not the rest.** Only the CARLA family takes
a ``t0``. The cfts baselines (Wachter/COMTE/CONFETI/CELS) have no intervention
timestep to pin: they edit the trajectory freely and their *derived*
``intervention_t`` is merely the first cell they happened to touch. On ``full``,
CftsWachter's median derived ``t0`` is **0** with ``validity = 0.99`` and
``sparsity = 0.02`` — it reaches the label by editing 98% of all ``T x k``
features with no mechanism propagation, not by intervening early. Plotting it
on a horizon axis would read as a refutation of H8 while measuring a different
object. The honest contrast is **constrained causal do() vs. unconstrained
edit**, and it is reported as a fixed reference line, not as a swept curve.

Run (after Phases 01-03 for the config)::

    uv run python experiments/08_horizon_sweep.py --config smoke --n-cf 10
    uv run python experiments/08_horizon_sweep.py --config full --horizons 2,5,10,25,50,75

Outputs under ``results/<config>/horizon/``:

* ``per_instance.csv`` — one row per (method, horizon, instance), the Phase-04
  schema plus ``horizon`` / ``t0``
* ``summary.json`` — per (method, horizon) aggregate, R7-stamped

**The model-vs-world audit over horizon.** Each sweep point also carries the
``PS`` (sufficiency) decomposition as ``ps_*`` columns: ``A`` (the model's
verdict on the method's proposed CF), ``B`` (the model's verdict on the *world's*
realisation of that same intervention), and ``C`` (the **world's** verdict on
it, read off the true SCM via ``scm_label`` — empirical, not a restatement of
the classifier). ``delta_outcome = B - C`` is model-vs-world on an identical
trajectory, so sweeping it says whether the model's causal claim degrades with
horizon *in the same way* the world's efficacy does, or comes apart from it.

This is **PS, not PNS.** ``X_sel`` is the flip-candidate set (non-target ->
target), so the estimand on it is sufficiency. The PN direction needs
target -> non-target counterfactuals regenerated at every horizon; until that
exists, a combined PNS is deliberately **not** synthesised from the missing
term (R3, matching Phase 07's default). Columns are prefixed ``ps_`` and
``summary.json`` records ``estimand: "PS"`` so the distinction survives contact
with a downstream reader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.methods import CARLARecourse, PearlCARLARecourse  # noqa: E402
from causaltemp_xai.metrics.pns import pns_direction, recover_label_threshold  # noqa: E402
from experiments._common import (  # noqa: E402
    aggregate_method_row,
    config_dir,
    dump_json,
    per_instance_records,
    set_run_context,
    write_csv,
)

#: Absolute horizons ``T - t0`` swept when ``--horizons`` is not given, as
#: fractions of ``T`` resolved to absolute steps at run time.
#:
#: Spacing is approximately **geometric**, because the quantity being resolved
#: decays geometrically: the Pearl delta measured on ``full`` falls from 1.0 at
#: ``t0`` to 4.5e-4 by ``T-1``, and on ``full_nl`` to 4.5e-9. A uniform grid
#: spends most of its points in the flat tail where every method has already
#: failed, and resolves the collapse knee — the only region that carries the
#: claim — with one or two points. The first validation run on ``full`` put the
#: knee between ``h = 10`` and ``h = 50``, and ``full_nl`` decays four orders
#: faster, so the knee moves left with the mechanism family; the grid has to be
#: fine at small ``h`` for both.
#:
#: Also brackets the locked default's operating point (``t0 in {25, 50}`` of
#: ``T = 100``, i.e. ``h in {75, 50}``) on both sides, so the committed tables
#: can be *located* on the curve rather than assumed to sit at its end.
DEFAULT_HORIZON_FRACTIONS = (0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.75, 0.9)


def resolve_horizons(T: int, spec: str | None) -> list[int]:
    """Absolute horizons to sweep for a series of length ``T``.

    ``spec`` is a comma-separated list of absolute horizons; ``None`` uses
    :data:`DEFAULT_HORIZON_FRACTIONS`. A horizon ``h`` means ``t0 = T - h``, and
    is kept only if that ``t0`` is scorable (``0 < t0 < T - 1``) — the same
    bound :func:`~causaltemp_xai.methods.counterfactual.carla._resolve_t0_candidates`
    enforces, applied here so an unusable request is reported before any
    optimisation runs rather than raising mid-sweep.
    """
    if spec:
        proposed = [int(h.strip()) for h in spec.split(",") if h.strip()]
    else:
        proposed = [max(1, round(f * T)) for f in DEFAULT_HORIZON_FRACTIONS]
    kept = sorted({h for h in proposed if 0 < T - h < T - 1})
    dropped = sorted(set(proposed) - set(kept))
    if dropped:
        print(
            f"[08] dropped unscorable horizons {dropped} for T={T} "
            f"(need 0 < T-h < T-1, i.e. 1 < h < {T})"
        )
    if not kept:
        raise SystemExit(f"[08] no scorable horizon in {proposed} for T={T}")
    return kept


def run(
    config_name: str,
    out_dir,
    horizons_spec: str | None = None,
    n_cf: int | None = None,
    seed: int | None = None,
) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    set_run_context(seed=cfg.seed, config=cfg.name)

    data = load_dataset(cfg.name, out_dir=out_dir)
    graph, mech = data["graph"], data["mechanism"]

    # World-side label threshold, recovered once from the full dataset. This is
    # what makes the PS ``C`` term empirical rather than a restatement of the
    # classifier: ``scm_label`` reads the *true* SCM trajectory, not the model.
    X_all = np.concatenate([data[f"X_{s}"] for s in ("train", "val", "test")])
    Y_all = np.concatenate([data[f"Y_{s}"] for s in ("train", "val", "test")])
    label = cfg.label_functional()
    theta = recover_label_threshold(X_all, Y_all, label_fn=label)

    res_dir = config_dir(cfg.name, "lstm")
    x_sel_path = res_dir / "cf" / "X_sel.npy"
    if not x_sel_path.exists():
        raise SystemExit(
            f"no {x_sel_path}; run first: uv run python experiments/03_run_cf_methods.py "
            f"--config {cfg.name}"
        )
    # Reuse Phase 03's selection so the sweep is comparable to the committed
    # tables instance-for-instance, not merely in distribution.
    X_sel = np.load(x_sel_path)
    if n_cf is not None:
        X_sel = X_sel[:n_cf]

    from causaltemp_xai.classifiers import LSTMClassifier

    clf = LSTMClassifier.load(Path(out_dir) / cfg.name / "lstm.pt")

    T = X_sel.shape[1]
    horizons = resolve_horizons(T, horizons_spec)
    t_label = label.label_site(T)
    print(f"[08] config={cfg.name} T={T} n_cf={len(X_sel)} " f"horizons={horizons} (t0 = T - h)")
    print(f"[08] label threshold theta={theta:+.6f} (world-side, for the PS C term)")
    # H8c (RISK-19): on the default terminal rule t_label == T-1 and the two
    # distances coincide, which is exactly why both are printed -- a horizon
    # curve is only attributable to `T - t0` if `t_label - t0` is also shown.
    print(f"[08] label functional={cfg.label_fn} site t_label={t_label} of T={T}")

    out_root = res_dir / "horizon"
    all_rows, summary = [], []

    hdr = (
        f"{'method':<14}{'h':>5}{'t0':>5}{'t_l-t0':>8}"
        f"{'valid':>8}{'roll_h':>8}{'pearl_h':>8}{'vac':>7}"
        f"{'A':>7}{'B':>7}{'C':>7}{'d_tot':>8}{'d_traj':>8}{'d_out':>8}{'n_ps':>6}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))

    for h in horizons:
        t0 = T - h
        builders = {
            # t0 bound as a default arg: without it the closure would read the
            # loop variable and every horizon would silently run at the last one.
            "CARLA": lambda t0=t0: CARLARecourse(target_class=1, n_steps=300, t0_steps=(t0,)),
            "PearlCARLA": lambda t0=t0: PearlCARLARecourse(target_class=1, t0_steps=(t0,)),
        }
        for name, build in builders.items():
            cfs = build().generate_batch(X_sel, clf, graph, mech)
            preds = np.asarray(clf.predict(cfs)).reshape(-1)
            rows = per_instance_records(cfg.name, "lstm", name, X_sel, cfs, graph, mech, preds)
            for r in rows:
                r["horizon"] = h
                r["t0"] = t0
                r["t_label"] = t_label
                r["label_horizon"] = t_label - t0
            all_rows.extend(rows)

            agg = aggregate_method_row(cfg.name, "lstm", name, rows)
            agg["horizon"] = h
            agg["t0"] = t0
            agg["t_label"] = t_label
            agg["label_horizon"] = t_label - t0

            # PS direction only. X_sel is the flip-candidate set (non-target ->
            # target), so this estimand is *sufficiency*, not PNS. The PN
            # direction needs target -> non-target CFs, which would have to be
            # generated afresh at every horizon; until they are, a combined PNS
            # is deliberately NOT synthesised from the missing term (R3, and
            # matching Phase 07's default). Keys are prefixed ``ps_`` so no
            # downstream reader can mistake this column for PNS.
            ps = pns_direction(X_sel, cfs, clf, mech, theta, target_class=1)
            agg.update({f"ps_{k}": v for k, v in ps.items()})
            summary.append(agg)

            def _f(v, w=8):
                return (
                    f"{v:>{w}.2f}" if v is not None and not np.isnan(v) else " " * (w - 3) + "N/A"
                )

            print(
                f"{name:<14}{h:>5}{t0:>5}{t_label - t0:>8}{_f(agg['validity'])}"
                f"{_f(agg['cf_faith_rollout_hard'])}{_f(agg['cf_faith_pearl_hard'])}"
                f"{_f(agg.get('frac_vacuous'), 7)}"
                f"{_f(ps['A_model_proposed'], 7)}{_f(ps['B_model_oracle'], 7)}"
                f"{_f(ps['C_world_oracle'], 7)}{_f(ps['delta_total'])}"
                f"{_f(ps['delta_trajectory'])}{_f(ps['delta_outcome'])}"
                f"{ps['n_scorable']:>6}"
            )

    write_csv(out_root / "per_instance.csv", all_rows)
    dump_json(
        out_root / "summary.json",
        {
            "T": T,
            "horizons": horizons,
            "label_threshold": theta,
            # Named so a reader cannot take the ps_* columns for PNS.
            "estimand": "PS",
            "PN_world": None,
            "PN_note": (
                "PN direction not run: it needs target -> non-target CFs generated "
                "at every horizon. Combined PNS is deliberately not synthesised "
                "from a missing term (R3, matching Phase 07's default)."
            ),
            "rows": summary,
        },
    )
    print(f"\n[08] wrote {out_root / 'per_instance.csv'}")
    print(f"[08] wrote {out_root / 'summary.json'}")
    print(
        "[08] read frac_vacuous alongside validity: a method whose delta collapses "
        "produces CFs that never intervened, which is a different failure from "
        "intervening and not reaching the label (RISK-17)."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--horizons",
        default=None,
        help="comma-separated absolute horizons T-t0 (default: fractions of T, "
        "denser near the collapse knee)",
    )
    parser.add_argument("--n-cf", type=int, default=None, help="truncate Phase 03's X_sel")
    parser.add_argument("--seed", type=int, default=None, help="seeded_variant replicate")
    args = parser.parse_args(argv)
    run(args.config, args.out_dir, args.horizons, args.n_cf, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
