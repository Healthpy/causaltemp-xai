"""Reproducible dataset generation, persistence, and loading for LinearSCM-T.

Datasets are deterministic given a :class:`~causaltemp_xai.config.BenchmarkConfig`
(the SCM seed fixes the graph, mechanisms, trajectories, and labels), so the
on-disk artifacts are **regenerated in CI** rather than committed — see the
plan's pinned decision and ``.gitignore`` (``data/linearscm_t/`` is ignored).

Layout written per config (``out_dir/<config.name>/``)::

    X_train.npy  X_val.npy  X_test.npy        # (n_split, T, k)
    Y_train.npy  Y_val.npy  Y_test.npy        # (n_split,)
    graph.npy                                  # (k, k, L)
    mechanisms.npz                             # A_0 … A_{L-1}, each (k, k)
    meta.json                                  # config + per-split class balance

CLI::

    uv run python -m causaltemp_xai.data_io --config smoke
    uv run python -m causaltemp_xai.data_io --config full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.config import (
    CONFIGS,
    BenchmarkConfig,
    get_config,
    shifted_config,
)

#: Default root directory for persisted datasets (gitignored).
DEFAULT_OUT_DIR = Path("data/linearscm_t")

#: Train/val/test fractions.
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


def stratified_split(
    Y: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint train/val/test index arrays, stratified on ``Y``.

    Splitting is performed per class so every class appears in every split at
    (approximately) the requested fractions. The shuffle is seeded for
    reproducibility.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * fractions[0]))
        n_val = int(round(n * fractions[1]))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])

    train = np.concatenate(train_idx)
    val = np.concatenate(val_idx)
    test = np.concatenate(test_idx)

    # Shuffle within each split so class blocks are interleaved.
    for split in (train, val, test):
        rng.shuffle(split)

    return train, val, test


def _class_balance(y: np.ndarray) -> dict[str, int]:
    """Return ``{class_label: count}`` as JSON-friendly string keys."""
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def generate_and_save(
    config: BenchmarkConfig,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> Path:
    """Generate a dataset from ``config`` and persist it to disk.

    Returns the directory the artifacts were written to
    (``out_dir/<config.name>/``).
    """
    out_dir = Path(out_dir)
    dest = out_dir / config.name
    dest.mkdir(parents=True, exist_ok=True)

    gen = LinearSCMT(
        k=config.k,
        L=config.L,
        sparsity=config.sparsity,
        noise_type=config.noise_type,
        T=config.T,
        N=config.N,
        seed=config.seed,
    )
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    graph, mechanisms = data["graph"], data["mechanisms"]

    train_idx, val_idx, test_idx = stratified_split(Y, seed=config.seed)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split_name, idx in splits.items():
        np.save(dest / f"X_{split_name}.npy", X[idx])
        np.save(dest / f"Y_{split_name}.npy", Y[idx])

    np.save(dest / "graph.npy", graph)
    np.savez(
        dest / "mechanisms.npz",
        **{f"A_{l}": A for l, A in enumerate(mechanisms)},
    )

    meta = {
        "config": config.as_dict(),
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_sizes": {name: int(len(idx)) for name, idx in splits.items()},
        "class_balance": {
            name: _class_balance(Y[idx]) for name, idx in splits.items()
        },
        "shapes": {
            "X_train": list(X[train_idx].shape),
            "graph": list(graph.shape),
        },
    }
    with open(dest / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return dest


def load_dataset(
    config_name: str,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> dict:
    """Load a previously persisted dataset.

    Returns a dict with keys ``X_train/X_val/X_test``, ``Y_train/Y_val/Y_test``,
    ``graph``, ``mechanisms`` (list of L arrays, ordered by lag), and ``meta``.
    """
    src = Path(out_dir) / config_name
    if not src.exists():
        raise FileNotFoundError(
            f"no dataset at {src}; run `python -m causaltemp_xai.data_io "
            f"--config {config_name}` first"
        )

    out: dict = {}
    for split_name in ("train", "val", "test"):
        out[f"X_{split_name}"] = np.load(src / f"X_{split_name}.npy")
        out[f"Y_{split_name}"] = np.load(src / f"Y_{split_name}.npy")

    out["graph"] = np.load(src / "graph.npy")

    with np.load(src / "mechanisms.npz") as mech:
        # Restore lag order: A_0, A_1, … A_{L-1}
        keys = sorted(mech.files, key=lambda s: int(s.split("_")[1]))
        out["mechanisms"] = [mech[k] for k in keys]

    with open(src / "meta.json") as fh:
        out["meta"] = json.load(fh)

    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and persist a LinearSCM-T dataset."
    )
    parser.add_argument(
        "--config",
        choices=sorted(CONFIGS),
        required=True,
        help="Named benchmark config to generate.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output root directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--shift-noise",
        choices=("laplace", "uniform"),
        default=None,
        help=(
            "Generate the Shift-VR environment instead: same SCM (graph/"
            "mechanisms) as --config but this innovation distribution. Saved "
            "under '<config>_shift'."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    config = get_config(args.config)
    if args.shift_noise is not None:
        config = shifted_config(config, noise_type=args.shift_noise)
    dest = generate_and_save(config, out_dir=args.out_dir)

    meta_path = dest / "meta.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    print(f"Wrote dataset for config '{config.name}' to {dest}")
    print(f"  split sizes:   {meta['split_sizes']}")
    print(f"  class balance: {meta['class_balance']}")


if __name__ == "__main__":
    main()
