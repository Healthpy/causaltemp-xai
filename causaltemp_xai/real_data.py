"""Loading for real (UEA/UCR) multivariate time series datasets (M4b).

M4b tests whether the horizon/decay result (H4, renumbered 2026-08-06, was H8)
reproduces on real signal,
with **no causal graph** -- discovered-graph CF-faith scoring is explicitly
out of scope (`ROADMAP.md` M4b DoD). This module is therefore a **parallel**
loader to :mod:`causaltemp_xai.data_io`, not a modification of it:
:func:`load_real_dataset` mirrors ``data_io.load_dataset()``'s
``X_train/val/test`` + ``Y_train/val/test`` shape contract but omits
``graph``/``mechanism`` entirely (``None``, not a placeholder) so downstream
code must explicitly branch on their absence rather than silently loading a
fake graph.

``.ts`` (sktime long-format) parsing is hand-rolled -- no sktime/aeon
dependency exists in this repo (`DECISIONS.md` 2026-08-05) and the format is
simple: header lines starting ``@key value`` (metadata) followed by
``@data``, then one line per instance::

    dim1_v1,dim1_v2,...,dim1_vT:dim2_v1,...,dim2_vT:...:dimK_v1,...,dimK_vT:label

-- ``K`` colon-separated blocks of ``T`` comma-separated floats (one block
per channel), followed by a final colon-separated string class label.
Verified directly against the downloaded UEA ``BasicMotions_TRAIN.ts``: 6
colon-blocks of 100 comma-values each, plus a trailing string label
(``"Standing"`` etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_REAL_DIR = Path("data/real")


def parse_ts_file(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one sktime ``.ts`` file into ``(X, labels)``.

    Returns
    -------
    X : ndarray, shape ``(N, T, k)``
        ``k`` = number of colon-separated dimension blocks, ``T`` = number of
        comma-separated values per block (equal-length series only --
        ``@equalLength true`` is not verified here, an unequal-length file
        raises ``ValueError`` from the ragged ``np.array`` construction).
    labels : ndarray of str, shape ``(N,)``
        The trailing class-label string per instance, unencoded.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"{path}: no '@data' marker found -- not a valid .ts file")

    rows_X: list[list[list[float]]] = []
    labels: list[str] = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 2:
            raise ValueError(f"{path}: data row has no label block: {line[:80]!r}")
        *dim_blocks, label = parts
        rows_X.append([[float(v) for v in block.split(",")] for block in dim_blocks])
        labels.append(label)

    X = np.asarray(rows_X, dtype=float)  # (N, k, T) -- dims are the outer axis in the file
    X = np.transpose(X, (0, 2, 1))  # -> (N, T, k), matching this repo's convention
    return X, np.asarray(labels)


def load_real_dataset(name: str, out_dir: Path | str = DEFAULT_REAL_DIR) -> dict:
    """Load a previously-prepared real dataset (see ``experiments/00_prepare_real_dataset.py``).

    Mirrors :func:`causaltemp_xai.data_io.load_dataset`'s return contract
    for the keys real data *can* supply, but with ``graph``/``mechanism``
    both ``None`` (never a placeholder) -- any downstream code touching them
    must branch explicitly, not silently proceed against a fake graph.

    Returns
    -------
    dict with keys ``X_train/X_val/X_test`` ``(n_split, T, k)``,
    ``Y_train/Y_val/Y_test`` ``(n_split,)`` (integer-encoded, per
    ``meta["class_names"]``), ``graph`` (``None``), ``mechanism`` (``None``),
    ``meta``.
    """
    src = Path(out_dir) / name
    if not src.exists():
        raise FileNotFoundError(
            f"no real dataset at {src}; run "
            f"`python experiments/00_prepare_real_dataset.py --name {name}` first"
        )

    out: dict = {}
    for split_name in ("train", "val", "test"):
        out[f"X_{split_name}"] = np.load(src / f"X_{split_name}.npy")
        out[f"Y_{split_name}"] = np.load(src / f"Y_{split_name}.npy")
    out["graph"] = None
    out["mechanism"] = None
    with open(src / "meta.json") as fh:
        out["meta"] = json.load(fh)
    return out
