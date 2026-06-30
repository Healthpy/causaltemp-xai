"""Fail-fast phenomenon check — the core scientific guard for the benchmark.

Before any IG / Shift-VR / figure work is built on top, this confirms the
hinge of the whole experiment on the smoke config: a causal recourse method
(CARLA) attains high CF-faith under the rollout semantics it is built to
satisfy, while a standard gradient CF (Wachter) does not. If that gap is
absent, the H1/H3 phenomenon has not reproduced and we STOP here rather than
discovering it after Stage 8.

Run::

    uv run python -m experiments.phenomenon_check --config smoke

Both CF-faith semantics are printed for both methods; only the *rollout*-metric
gap (the one CARLA is constructed to satisfy) gates the STOP decision. The
``pearl`` metric is expected ≈0 for CARLA too — that is the documented
rollout-vs-pearl contrast, not a phenomenon failure.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.config import get_config
from causaltemp_xai.eval import evaluate_method
from causaltemp_xai.methods import CARLARecourse, WachterCF, derive_intervention_t


def _train_smoke_classifier(X, Y, k, seed=0):
    """Lightly-trained LSTM — accuracy is irrelevant to the construction-level
    CF-faith gap, so a small/fast model suffices for the guard."""
    n = len(X)
    n_tr = int(n * 0.8)
    clf = LSTMClassifier(
        n_inputs=k,
        hidden_size=32,
        num_layers=2,
        dropout=0.0,
        lr=3e-3,
        batch_size=64,
        max_epochs=40,
        patience=10,
        seed=seed,
    )
    clf.fit(X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:])
    return clf, n_tr


def run(config_name: str = "smoke", n_instances: int = 10) -> dict:
    cfg = get_config(config_name)
    gen = LinearSCMT(
        k=cfg.k,
        L=cfg.L,
        sparsity=cfg.sparsity,
        noise_type=cfg.noise_type,
        T=cfg.T,
        N=cfg.N,
        seed=cfg.seed,
    )
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    graph, mech = data["graph"], data["mechanism"]

    clf, n_tr = _train_smoke_classifier(X, Y, cfg.k, seed=cfg.seed)
    X_train = X[:n_tr]

    # Select instances the classifier currently predicts as class 0; flip to 1.
    test_X = X[n_tr:]
    preds = clf.predict(test_X)
    src = [i for i in range(len(test_X)) if preds[i] == 0]
    if len(src) < n_instances:
        src = list(range(len(test_X)))
    src = src[:n_instances]
    X_sel = test_X[src]

    print(f"Config '{config_name}': k={cfg.k} T={cfg.T} N={cfg.N}")
    print(f"  selected {len(X_sel)} instances predicted as class 0\n")

    wachter = WachterCF(target_class=1, n_steps=300, lr=0.1)
    carla = CARLARecourse(target_class=1, n_steps=300, t0_fractions=(0.25, 0.5))

    wachter_cfs = wachter.generate_batch(X_sel, clf)
    carla_cfs = carla.generate_batch(X_sel, clf, graph, mech)

    w = evaluate_method(clf, X_sel, wachter_cfs, X_train, graph, mech, target_class=1)
    c = evaluate_method(clf, X_sel, carla_cfs, X_train, graph, mech, target_class=1)

    w_t0 = [derive_intervention_t(X_sel[i], wachter_cfs[i]) for i in range(len(X_sel))]
    c_t0 = [derive_intervention_t(X_sel[i], carla_cfs[i]) for i in range(len(X_sel))]

    def _row(name, m):
        return (
            f"  {name:8s}  validity={m['validity']:.2f}  "
            f"rollout_hard={m['cf_faith_rollout_hard']:.2f}  "
            f"rollout_soft={m['cf_faith_rollout_soft']:.2f}  "
            f"pearl_hard={m['cf_faith_pearl_hard']:.2f}  "
            f"pearl_soft={m['cf_faith_pearl_soft']:.2f}"
        )

    print("CF-faith (both semantics):")
    print(_row("Wachter", w))
    print(_row("CARLA", c))
    print()
    print(f"intervention_t  Wachter: {w_t0}")
    print(f"intervention_t  CARLA  : {c_t0}")
    print(f"  (T-1 = {cfg.T - 1}; Wachter clustered at T-1 would be a red flag)\n")

    gap = c["cf_faith_rollout_hard"] - w["cf_faith_rollout_hard"]
    wachter_at_end = float(np.mean(np.asarray(w_t0) == cfg.T - 1))

    phenomenon_holds = c["cf_faith_rollout_hard"] > w["cf_faith_rollout_hard"]
    print(
        f"PHENOMENON {'OK' if phenomenon_holds else 'ABSENT'}: "
        f"CARLA rollout_hard ({c['cf_faith_rollout_hard']:.2f}) "
        f"{'>' if phenomenon_holds else '<='} "
        f"Wachter rollout_hard ({w['cf_faith_rollout_hard']:.2f}); "
        f"gap={gap:+.2f}, Wachter@T-1 frac={wachter_at_end:.2f}"
    )

    return {
        "wachter": w,
        "carla": c,
        "wachter_t0": w_t0,
        "carla_t0": c_t0,
        "gap": gap,
        "phenomenon_holds": phenomenon_holds,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fail-fast CF-faith phenomenon check.")
    parser.add_argument("--config", default="smoke", help="Named benchmark config.")
    parser.add_argument("--n-instances", type=int, default=10)
    args = parser.parse_args(argv)

    result = run(args.config, n_instances=args.n_instances)
    if not result["phenomenon_holds"]:
        print(
            "\nSTOP: the rollout CF-faith gap is absent. Per the MVP risk table "
            "this is a documentable outcome — record it in the plan Issues and "
            "surface to the user before building Stages 6–8.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
