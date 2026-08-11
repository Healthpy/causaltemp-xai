"""Phase 11: Tier-3 orchestration scaffold (M4i, `DECISIONS.md` 2026-08-06).

A **dataset-agnostic, graph-source-agnostic** orchestrator for a real dataset
that may or may not have a known causal graph -- "ground truth if known can be
used, otherwise causal discovery methods or/and domain knowledge derived SCM
and causal graph can be used."

**Explicit scope boundary.** This does **not** commit to SepsisSim or
MIMIC-IV. `ROADMAP.md` M5 (SepsisSim go/no-go) remains genuinely undecided --
no PI memo, no confirmed data access, zero existing sepsis code or data
anywhere in this repo. Per `CLAUDE.md`'s standing rule ("Adding or removing
any method, metric, experiment phase, or publication target requires a
recorded PI decision in `ROADMAP.md` **before** code is written"), no code
here assumes MIMIC-IV/SepsisSim access is real. This file becomes
SepsisSim-ready only once/if M5 is later resolved -- building it is not
itself a Go/No-Go decision and does not pre-empt one.

**Every tier generates its own counterfactual explanations -- Tier 3 must not
merely assume Tier 2 already ran for whatever dataset it's pointed at.**
Before any mode-specific logic, `dispatch` ensures CFs are persisted for
`--name` by calling `06b_horizon_sweep_real.run` (the same generic real-data
CF-generation phase Tier 2 uses: trains an LSTM, runs the 5 graph-free CF
methods) if they are not already there -- mirroring Tier 2's own `--skip-cf`
convention exactly, so re-running against an already-processed dataset does
not regenerate for no reason.

Three graph-source modes (`--mode`):

- **`known`** -- a graph (and optionally a mechanism) is supplied directly, no
  discovery needed. Bare graph: a `.npy` `(k,k,L)` adjacency. Mechanism (if
  given): a `.npz` with keys `A0..A{L-1}`, one `(k,k)` array per lag,
  reconstructed as a `LinearMechanism`. If a mechanism is supplied, the
  freshly-generated (or already-persisted) CFs are scored against it directly
  (`experiments.07_auxiliary_methods._mean_soft_cf_faith`) -- nothing is fit,
  since the mechanism is given, not discovered. Graph-only: records the
  structural facts, notes CF-faith is not computable without a mechanism.
- **`discover`** -- reuses Tier 2's exact real-data path: DYNOTEARS +
  PCMCIplus fit on `--name`'s data
  (`experiments.07b_discovered_graph_real.run_cross_method_agreement_real`,
  structural agreement only, no ground truth). With
  `--with-mechanism-cf-faith`, also builds a DYNOTEARS `to_linear_mechanism()`
  bootstrap ensemble and scores the generated CFs against it
  (`07b._fit_mechanism_ensemble` + `07b.score_method_discovered`) -- the exact
  M4g pattern, applied generically, not sepsis-specific.
- **`domain`** -- a hand-built mechanism is supplied directly (`.npz`,
  bypassing a bare graph file); the graph is derived as `(A != 0)` per lag;
  otherwise identical to `known` + mechanism.

**Smoke-test path, since no real Tier-3 dataset exists.** `discover` mode
against an already-wired Tier-2 dataset (no known graph exists for those
either) demonstrates the machinery works. Every artifact from such a run
carries an explicit note that this is **not** a Tier-3 substance claim --
`known`/`domain` are exercised as deterministic unit tests instead (dumping
one Tier-1 config's own true graph/mechanism into this file's serialization
format), which need no network and are fully reproducible.

Writes under `results/real_<name>/tier3/<mode>/`, a namespace distinct from
Tier 2's own `results/real_<name>/{dynotears,cross_method_agreement}/`
artifacts for the same `--name`, so a `discover`-mode smoke test never
collides with or overwrites those. The one shared subtree is
`results/real_<name>/lstm/cf/` -- deliberately: it is the CF-generation output
itself (`06b_horizon_sweep_real.run`), and Tier 2 and Tier 3 use the exact
same generic phase to produce it, so a dataset already processed by one tier
does not need to be regenerated for the other (`--skip-cf` short-circuits
when it's already there).

Usage
-----
    # First run for a dataset: generates CFs (06b) then applies discover mode.
    uv run python experiments/11_tier3_real_suite.py --name epilepsy --mode discover --with-mechanism-cf-faith
    # Already processed (by this file or by the Tier-2 suite): skip regeneration.
    uv run python experiments/11_tier3_real_suite.py --name basicmotions --mode discover --with-mechanism-cf-faith --skip-cf
    uv run python experiments/11_tier3_real_suite.py --name <dataset> --mode known --graph-file G.npy [--mechanism-file M.npz]
    uv run python experiments/11_tier3_real_suite.py --name <dataset> --mode domain --mechanism-file M.npz
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism  # noqa: E402
from causaltemp_xai.real_data import DEFAULT_REAL_DIR  # noqa: E402
from experiments._common import RESULTS_DIR, dump_json  # noqa: E402

_phase06b = importlib.import_module("experiments.06b_horizon_sweep_real")
_phase07 = importlib.import_module("experiments.07_auxiliary_methods")
_phase07b = importlib.import_module("experiments.07b_discovered_graph_real")

STAND_IN_NOTE = (
    "Tier-3 machinery demonstrated against a Tier-2 dataset (no known graph "
    "exists for it either); this is NOT a Tier-3 substance claim."
)


# ---------------------------------------------------------------------------
# Minimal graph/mechanism serialization (nothing like this existed before)
# ---------------------------------------------------------------------------


def save_mechanism_npz(mechanism: LinearMechanism, path) -> None:
    """Dump a `LinearMechanism`'s per-lag coefficient matrices to a `.npz`
    with keys `A0..A{L-1}`. Used by tests to construct `known`/`domain`
    fixtures deterministically from a Tier-1 config's own true mechanism."""
    kwargs = {f"A{l}": mechanism.A_list[l] for l in range(mechanism.L)}
    np.savez(path, **kwargs)


def load_mechanism_npz(path) -> LinearMechanism:
    data = np.load(path)
    keys = sorted(data.files, key=lambda k: int(k[1:]))
    return LinearMechanism([data[k] for k in keys])


def load_graph_npy(path) -> np.ndarray:
    return np.load(path)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


def _cf_dir(name: str, out_dir) -> Path:
    return ROOT / "results" / f"real_{name}" / "lstm" / "cf"


def _ensure_cfs(name: str, out_dir, n_cf, target_class: int, seed: int, skip_cf: bool) -> None:
    """Generate counterfactual explanations for `--name` (via the same
    generic real-data pipeline Tier 2 uses) unless already persisted and
    `skip_cf` is set. Every tier generates its own CFs -- Tier 3 does not
    assume some other phase already ran for whatever dataset it's pointed at.
    """
    cf_sel_path = _cf_dir(name, out_dir) / "X_sel.npy"
    if skip_cf and cf_sel_path.exists():
        print(f"[11] --skip-cf: reusing existing persisted CF arrays for {name!r}")
        return
    print(f"[11] generating counterfactual explanations for {name!r} (06b) ...")
    _phase06b.run(name=name, out_dir=out_dir, n_cf=n_cf, target_class=target_class, seed=seed)


def _score_against_mechanism(name: str, out_dir, mechanism: LinearMechanism) -> dict | None:
    """Score already-persisted CF arrays against a given (not fitted)
    mechanism, both semantics. Returns `None` if no CF arrays are persisted
    for `--name` yet."""
    from causaltemp_xai.metrics.cf_faith import CFfaith

    cf_dir = _cf_dir(name, out_dir)
    x_sel_path = cf_dir / "X_sel.npy"
    if not x_sel_path.exists():
        return None
    X_sel = np.load(x_sel_path)
    k, L = mechanism.k, mechanism.L
    graph_placeholder = np.ones((k, k, L))
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")

    rows = {}
    for cf_path in sorted(cf_dir.glob("X_cf_*.npy")):
        method = cf_path.stem[len("X_cf_") :]
        cfs = np.load(cf_path)
        rows[method] = {
            "cf_faith_rollout": _phase07._mean_soft_cf_faith(
                X_sel, cfs, graph_placeholder, mechanism, rollout
            ),
            "cf_faith_pearl": _phase07._mean_soft_cf_faith(
                X_sel, cfs, graph_placeholder, mechanism, pearl
            ),
        }
    return rows


def run_known(
    name: str,
    out_dir,
    graph_file,
    mechanism_file=None,
    n_cf=None,
    target_class: int = 0,
    seed: int = 0,
    skip_cf: bool = False,
) -> None:
    _ensure_cfs(name, out_dir, n_cf, target_class, seed, skip_cf)
    graph = load_graph_npy(graph_file)
    out = {
        "provenance": {"dataset": name, "mode": "known", "graph_file": str(graph_file)},
        "graph_edges": int(graph.sum()),
        "graph_shape": list(graph.shape),
    }
    if mechanism_file is not None:
        mechanism = load_mechanism_npz(mechanism_file)
        out["provenance"]["mechanism_file"] = str(mechanism_file)
        cf_faith = _score_against_mechanism(name, out_dir, mechanism)
        if cf_faith is None:
            out["cf_faith_note"] = (
                f"No CF arrays persisted for {name!r} yet -- run "
                "experiments/06b_horizon_sweep_real.py (or the Tier-2 suite) first."
            )
        else:
            out["cf_faith_by_method"] = cf_faith
    else:
        out["cf_faith_note"] = (
            "No mechanism supplied -- CF-faith is not computable from a bare graph."
        )

    out_dir_res = ROOT / "results" / f"real_{name}" / "tier3" / "known"
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_agreement.json", out)
    print(f"[11] wrote {out_dir_res / 'graph_agreement.json'}")


def run_domain(
    name: str,
    out_dir,
    mechanism_file,
    n_cf=None,
    target_class: int = 0,
    seed: int = 0,
    skip_cf: bool = False,
) -> None:
    _ensure_cfs(name, out_dir, n_cf, target_class, seed, skip_cf)
    mechanism = load_mechanism_npz(mechanism_file)
    graph = np.stack([(A != 0).astype(int) for A in mechanism.A_list], axis=-1)
    out = {
        "provenance": {"dataset": name, "mode": "domain", "mechanism_file": str(mechanism_file)},
        "graph_edges": int(graph.sum()),
        "graph_shape": list(graph.shape),
    }
    cf_faith = _score_against_mechanism(name, out_dir, mechanism)
    if cf_faith is None:
        out["cf_faith_note"] = (
            f"No CF arrays persisted for {name!r} yet -- run "
            "experiments/06b_horizon_sweep_real.py (or the Tier-2 suite) first."
        )
    else:
        out["cf_faith_by_method"] = cf_faith

    out_dir_res = ROOT / "results" / f"real_{name}" / "tier3" / "domain"
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_agreement.json", out)
    print(f"[11] wrote {out_dir_res / 'graph_agreement.json'}")


def run_discover(
    name: str,
    out_dir,
    ensemble_b: int = 5,
    pc_alpha: float = 0.05,
    threshold: float = 0.1,
    seed: int = 0,
    with_mechanism_cf_faith: bool = True,
    stand_in: bool = True,
    n_cf=None,
    target_class: int = 0,
    skip_cf: bool = False,
) -> None:
    """Reuses Tier 2's exact real-data path: `run_cross_method_agreement_real`
    for the graph-level check, optionally `_fit_mechanism_ensemble` +
    `score_method_discovered` (M4g's own functions) for mechanism-level
    CF-faith -- the same pattern M4g built for Tier 2, applied here
    generically rather than hardcoded to a specific dataset.
    """
    from causaltemp_xai.real_data import load_real_dataset

    _ensure_cfs(name, out_dir, n_cf, target_class, seed, skip_cf)
    _phase07b.run_cross_method_agreement_real(name, out_dir, pc_alpha, threshold, seed)
    cma_path = ROOT / "results" / f"real_{name}" / "cross_method_agreement" / "graph_agreement.json"
    with open(cma_path) as fh:
        cma = json.load(fh)

    out = {
        "provenance": {
            "dataset": name,
            "mode": "discover",
            "ensemble_b": ensemble_b if with_mechanism_cf_faith else 0,
            "pc_alpha": pc_alpha,
            "threshold": threshold,
        },
        "shd_between_methods": cma["shd_between_methods"],
        "graph_agreement_note": cma["note"],
    }
    if stand_in:
        out["stand_in_note"] = STAND_IN_NOTE

    out_dir_res = ROOT / "results" / f"real_{name}" / "tier3" / "discover"
    out_dir_res.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir_res / "graph_agreement.json", out)
    print(f"[11] wrote {out_dir_res / 'graph_agreement.json'}")
    if stand_in:
        print(f"[11] NOTE: {STAND_IN_NOTE}")

    if with_mechanism_cf_faith:
        cf_dir = _cf_dir(name, out_dir)
        # _ensure_cfs above guarantees these are persisted by this point.
        data = load_real_dataset(name, out_dir=out_dir)
        X_all = np.concatenate([data["X_train"], data["X_val"]], axis=0)
        k = X_all.shape[-1]
        mechanisms = _phase07b._fit_mechanism_ensemble(
            X_all, k, 1, ensemble_b, seed=seed, threshold=threshold
        )
        X_sel = np.load(cf_dir / "X_sel.npy")
        method_rows = []
        for cf_path in sorted(cf_dir.glob("X_cf_*.npy")):
            method = cf_path.stem[len("X_cf_") :]
            cfs = np.load(cf_path)
            row = _phase07b.score_method_discovered(X_sel, cfs, mechanisms, seed=seed)
            method_rows.append({"method": method, **row})

        cf_faith_out = {
            "provenance": out["provenance"],
            "methods": method_rows,
            "limitation": _phase07b.LIMITATION_NOTE,
        }
        if stand_in:
            cf_faith_out["stand_in_note"] = STAND_IN_NOTE
        dump_json(out_dir_res / "cf_faith_discovered.json", cf_faith_out)
        print(f"[11] wrote {out_dir_res / 'cf_faith_discovered.json'}")


def dispatch(
    mode: str,
    name: str,
    out_dir,
    graph_file=None,
    mechanism_file=None,
    ensemble_b: int = 5,
    pc_alpha: float = 0.05,
    threshold: float = 0.1,
    seed: int = 0,
    with_mechanism_cf_faith: bool = True,
    n_cf=None,
    target_class: int = 0,
    skip_cf: bool = False,
) -> None:
    if mode == "known":
        if graph_file is None:
            raise SystemExit("[11] --mode known requires --graph-file")
        run_known(
            name,
            out_dir,
            graph_file,
            mechanism_file=mechanism_file,
            n_cf=n_cf,
            target_class=target_class,
            seed=seed,
            skip_cf=skip_cf,
        )
    elif mode == "domain":
        if mechanism_file is None:
            raise SystemExit("[11] --mode domain requires --mechanism-file")
        run_domain(
            name,
            out_dir,
            mechanism_file,
            n_cf=n_cf,
            target_class=target_class,
            seed=seed,
            skip_cf=skip_cf,
        )
    elif mode == "discover":
        run_discover(
            name,
            out_dir,
            ensemble_b=ensemble_b,
            pc_alpha=pc_alpha,
            threshold=threshold,
            seed=seed,
            with_mechanism_cf_faith=with_mechanism_cf_faith,
            n_cf=n_cf,
            target_class=target_class,
            skip_cf=skip_cf,
        )
    else:
        raise SystemExit(f"[11] unknown --mode {mode!r}")


def aggregate_tier3_suite(out_root=None) -> None:
    """Cross-invocation manifest: scans every `results/real_*/tier3/*/
    graph_agreement.json` written so far (Tier 3 runs one `--name`/`--mode`
    per invocation, unlike 09/10's loops, so this aggregate accumulates
    across separate runs rather than summarizing a single one)."""
    out_root = out_root or (RESULTS_DIR / "tier3_suite")
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(
        glob.glob(str(RESULTS_DIR / "real_*" / "tier3" / "*" / "graph_agreement.json"))
    ):
        with open(path) as fh:
            data = json.load(fh)
        records.append({"path": path, **data.get("provenance", {})})
    dump_json(out_root / "summary.json", {"runs": records})
    print(f"[11] wrote {out_root / 'summary.json'} ({len(records)} runs)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True, choices=("known", "discover", "domain"))
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--mechanism-file", default=None)
    parser.add_argument("--ensemble-b", type=int, default=5)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--with-mechanism-cf-faith", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument(
        "--n-cf", type=int, default=None, help="CF-generation instance count, forwarded to 06b."
    )
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument(
        "--skip-cf",
        action="store_true",
        help="Skip CF generation if already persisted for --name (mirrors Tier 2's own flag).",
    )
    args = parser.parse_args(argv)

    dispatch(
        args.mode,
        args.name,
        args.out_dir,
        graph_file=args.graph_file,
        mechanism_file=args.mechanism_file,
        ensemble_b=args.ensemble_b,
        pc_alpha=args.pc_alpha,
        threshold=args.threshold,
        seed=args.seed,
        with_mechanism_cf_faith=args.with_mechanism_cf_faith,
        n_cf=args.n_cf,
        target_class=args.target_class,
        skip_cf=args.skip_cf,
    )
    aggregate_tier3_suite()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
