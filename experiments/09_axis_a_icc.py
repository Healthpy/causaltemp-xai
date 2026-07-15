"""Phase 09: decoder-based Axis-A ICC run (latent-traversal Interventional
Concept Consistency), the definitional ICC for representation methods.

Trains an **iVAE** (a method that exposes an encoder + decoder) on a config's
observational data, then scores each latent dimension with
:func:`causaltemp_xai.metrics.axis_a.icc_latent` (matched-baseline, ±δ,
std-scaled), against the frozen LSTM. Because iVAE identifies factors only up to
permutation, latent dims are first **aligned to ground-truth channels** via a
Hungarian match on the |correlation| matrix; the causally-relevant concepts are
the label channel (channel 0, since ``Y = 1[X_T^0 > θ]``) and its graph
ancestors. ICC is then contrasted between causal-relevant-aligned and
non-relevant-aligned latent dims.

Per the PI metric review, this run reports the **sanity gates first** — if the
iVAE reconstruction does not preserve the classifier's decision
(``recon_label_agreement`` low), ICC is *not interpretable* and that is reported
honestly rather than forced. Magnitudes ``c ∈ {1,2,3}`` are swept (pre-registered;
no δ-hacking), and a permuted-assignment null is reported.

Writes ``results/<config>/ivae/icc.json``.

Usage
-----
    uv run python experiments/09_axis_a_icc.py --config smoke
    uv run python experiments/09_axis_a_icc.py --config smoke_nl --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, load_dataset  # noqa: E402
from causaltemp_xai.methods.concept import iVAE  # noqa: E402
from causaltemp_xai.metrics.axis_a import icc_latent  # noqa: E402
from experiments._common import config_dir, dump_json  # noqa: E402

RECON_AGREEMENT_GATE = 0.8  # below this, ICC is not interpretable


def _ancestors_of(node: int, graph: np.ndarray) -> set[int]:
    """Transitive causal ancestors of ``node`` (incl. itself) in the lagged
    graph ``(k, k, L)`` where ``graph[i, j, l]==1`` means j causes i."""
    parents = lambda i: {j for j in range(graph.shape[1]) if np.any(graph[i, j, :] != 0)}
    seen, stack = {node}, [node]
    while stack:
        cur = stack.pop()
        for p in parents(cur):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _align_latents_to_channels(Z: np.ndarray, F: np.ndarray):
    """Hungarian match on |Pearson corr| between latent dims ``Z`` (N, d_z) and
    ground-truth factors ``F`` (N, k). Returns (assign, mcc): ``assign[i]`` =
    channel matched to latent dim ``i``; ``mcc`` = mean matched |corr|."""
    d_z, k = Z.shape[1], F.shape[1]
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Fc = (F - F.mean(0)) / (F.std(0) + 1e-12)
    corr = np.abs(Zc.T @ Fc) / Z.shape[0]  # (d_z, k)
    rows, cols = linear_sum_assignment(-corr)
    assign = {int(r): int(c) for r, c in zip(rows, cols)}
    mcc = float(corr[rows, cols].mean())
    return assign, mcc


def run(config_name: str, out_dir, epochs: int = 50, seed: int | None = None) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    data = load_dataset(cfg.name, out_dir=out_dir)
    graph = data["graph"]
    k = graph.shape[0]
    X_train, X_test = data["X_train"], data["X_test"]

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"[09] no classifier at {ckpt}; run experiments/02_train_classifiers.py "
            f"--config {cfg.name} first."
        )
    clf = LSTMClassifier.load(ckpt)

    # Train iVAE with latent_dim = k so dims align one-to-one with channels.
    print(f"[09] training iVAE (latent_dim={k}, epochs={epochs}) on {X_train.shape[0]} seqs ...")
    ae = iVAE(latent_dim=k, n_segments=min(4, k), epochs=epochs,
              beta=0.3, kl_warmup_frac=0.3)
    ae.fit_unsupervised(X_train)

    # --- Sanity gates (report FIRST) ---
    Z = ae.encode(X_test)                         # (N, k)
    X_recon = ae.decode(Z)                         # (N, T, k)
    recon_mse = float(np.mean((X_recon - X_test) ** 2))
    f_x = np.asarray(clf.predict(X_test)).reshape(-1)
    f_recon = np.asarray(clf.predict(X_recon)).reshape(-1)
    recon_label_agreement = float(np.mean(f_recon == f_x))

    # Ground-truth factors = per-channel final values (the label-driving repr).
    F = X_test[:, -1, :]                           # (N, k)
    assign, mcc_val = _align_latents_to_channels(Z, F)

    relevant_channels = _ancestors_of(0, graph)    # label channel 0 + ancestors
    parent_dims = [i for i in range(k) if assign.get(i) in relevant_channels]
    nonparent_dims = [i for i in range(k) if assign.get(i) not in relevant_channels]

    interpretable = recon_label_agreement >= RECON_AGREEMENT_GATE
    print(f"[09] GATES: recon_mse={recon_mse:.4f}  recon_label_agreement={recon_label_agreement:.2f} "
          f"(gate>={RECON_AGREEMENT_GATE})  MCC(latent,channel)={mcc_val:.2f}")
    print(f"[09] label-relevant channels (anc. of 0): {sorted(relevant_channels)}; "
          f"latent->channel assign: {assign}")
    if not interpretable:
        print("[09] recon_label_agreement below gate -> ICC is NOT interpretable on this "
              "run (iVAE reconstruction does not preserve the classifier's decision). "
              "Reporting gates only; this is itself an honest Axis-A finding.")

    # --- ICC magnitude ladder (pre-registered c in {1,2,3}) ---
    rng = np.random.default_rng(cfg.seed)
    ladder = []
    for c in (1.0, 2.0, 3.0):
        icc = icc_latent(X_test, ae.encode, ae.decode, clf, delta=c,
                         scale_by_std=True, symmetric=True)  # (k,)
        parent_mean = float(np.mean([icc[i] for i in parent_dims])) if parent_dims else float("nan")
        nonparent_mean = float(np.mean([icc[i] for i in nonparent_dims])) if nonparent_dims else float("nan")
        # Permuted-assignment null: random partition of the same size as parent_dims.
        perm = rng.permutation(k)
        null_parent = perm[:len(parent_dims)]
        null_mean = float(np.mean([icc[i] for i in null_parent])) if len(null_parent) else float("nan")
        ladder.append({
            "c": c,
            "icc_per_dim": [float(v) for v in icc],
            "parent_aligned_mean": parent_mean,
            "nonparent_aligned_mean": nonparent_mean,
            "contrast": (parent_mean - nonparent_mean),
            "permuted_null_mean": null_mean,
        })
        print(f"[09] c={c:g}: ICC parent-aligned={parent_mean:.3f}  non-parent={nonparent_mean:.3f}  "
              f"contrast={parent_mean - nonparent_mean:+.3f}  (null={null_mean:.3f})")

    out = {
        "provenance": {
            "config": cfg.as_dict(), "method": "iVAE", "metric": "icc_latent",
            "n_eval": int(X_test.shape[0]), "latent_dim": k, "epochs": epochs,
            "note": "matched-baseline f(D(z)); +/-delta; delta_i=c*std(z_i); "
                    "latents aligned to channels via Hungarian on |corr|.",
        },
        "gates": {
            "recon_mse": recon_mse,
            "recon_label_agreement": recon_label_agreement,
            "interpretable": interpretable,
            "mcc_latent_channel": mcc_val,
        },
        "alignment": {
            "latent_to_channel": assign,
            "label_relevant_channels": sorted(int(c) for c in relevant_channels),
            "parent_aligned_dims": parent_dims,
            "nonparent_aligned_dims": nonparent_dims,
        },
        "icc_ladder": ladder,
    }
    res_dir = config_dir(cfg.name, "ivae")
    res_dir.mkdir(parents=True, exist_ok=True)
    dump_json(res_dir / "icc.json", out)
    print(f"[09] wrote {res_dir / 'icc.json'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Decoder-based Axis-A ICC (iVAE latent traversal).")
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=50, help="iVAE training epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Multi-seed replicate: override the config seed via seeded_variant.")
    args = parser.parse_args(argv)
    run(args.config, args.out_dir, epochs=args.epochs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
