"""Phase 02: Train the LSTM classifier on an SCM-T benchmark.

Training is mechanism-agnostic (the LSTM only sees ``(X, Y)`` arrays), so any
registered config -- linear or nonlinear (``smoke_nl``, ``full_nl``) -- can be
trained here. ``--all`` defaults to the linear configs only, since the
nonlinear benchmark's *primary* evaluation path is the classifier-free oracle
structural-CF in Phase 05 (see ``docs/general_plan.md`` §6) -- but pass ``--config smoke_nl`` explicitly to also get a classifier for
it (e.g. to later explore real CF methods / attribution axes on nonlinear
data, same as the linear pipeline).

Checkpoints are written to ``data/scm_t/<config>/lstm.pt`` (same location the
downstream phases already expect), and a small accuracy report is written to
``results/<config>/lstm/train_report.json`` for the tables/figures phases.

Usage
-----
    uv run python experiments/02_train_classifiers.py --config smoke
    uv run python experiments/02_train_classifiers.py --config smoke_nl
    uv run python experiments/02_train_classifiers.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import CONFIGS, get_config, seeded_variant  # noqa: E402
from causaltemp_xai.data_io import DEFAULT_OUT_DIR, generate_and_save, load_dataset  # noqa: E402
from experiments._common import config_dir, dump_json, set_run_context  # noqa: E402

LINEAR_CONFIGS = ["smoke", "full", "full_sparse"]
TARGET_ACC = 0.90


def train_one(config_name: str, out_dir, seed: int | None = None, **hparams) -> None:
    cfg = get_config(config_name)
    if seed is not None:
        cfg = seeded_variant(cfg, seed)
    set_run_context(seed=cfg.seed, config=cfg.name)
    out_dir = Path(out_dir)
    if not (out_dir / cfg.name / "meta.json").exists():
        print(f"[02] dataset for '{cfg.name}' not found -- generating.")
        generate_and_save(cfg, out_dir=out_dir)
    data = load_dataset(cfg.name, out_dir=out_dir)

    clf = LSTMClassifier(n_inputs=cfg.k, seed=cfg.seed, **hparams)
    print(f"[02] training LSTM on '{cfg.name}' ...")
    clf.fit(data["X_train"], data["Y_train"], data["X_val"], data["Y_val"])

    accuracies = {
        "train": clf.score(data["X_train"], data["Y_train"]),
        "val": clf.score(data["X_val"], data["Y_val"]),
        "test": clf.score(data["X_test"], data["Y_test"]),
    }
    print(
        f"     train_acc={accuracies['train']:.4f} val_acc={accuracies['val']:.4f} "
        f"test_acc={accuracies['test']:.4f}"
    )
    if accuracies["test"] < TARGET_ACC:
        print(f"     WARNING: test accuracy {accuracies['test']:.3f} < target {TARGET_ACC:.2f}")

    ckpt_path = out_dir / cfg.name / "lstm.pt"
    clf.save(ckpt_path)
    print(f"     checkpoint -> {ckpt_path}")

    report_path = config_dir(cfg.name, "lstm") / "train_report.json"
    dump_json(report_path, {"config": cfg.name, "accuracies": accuracies, "target_acc": TARGET_ACC})
    print(f"     report -> {report_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the LSTM classifier for an SCM-T benchmark."
    )
    parser.add_argument("--config", choices=sorted(CONFIGS), default=None)
    parser.add_argument("--all", action="store_true", help="Train on every linear config.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Multi-seed replicate (M2): override --config's registered seed "
            "via causaltemp_xai.config.seeded_variant (dataset + classifier "
            "both re-seeded), reading/writing under '<config>_seed<seed>'. "
            "Ignored with --all."
        ),
    )
    args = parser.parse_args(argv)

    if not args.all and args.config is None:
        parser.error("pass --config <name> or --all")

    hparams = dict(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    names = LINEAR_CONFIGS if args.all else [args.config]
    for name in names:
        train_one(name, args.out_dir, seed=args.seed if not args.all else None, **hparams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
