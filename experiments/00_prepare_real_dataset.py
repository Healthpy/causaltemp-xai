"""Phase 00: Download and cache a real UEA/UCR dataset for M4b.

One-time preparation step for M4b's real-signal horizon external validity
check: fetches a UEA archive dataset's ``.ts`` files (stdlib
``urllib.request``, no new HTTP dependency), parses them with
``causaltemp_xai.real_data.parse_ts_file``, splits the archive's TRAIN split
into train/val (stratified, archive TEST stays held out as this dataset's
test split), integer-encodes labels, and writes the result under
``data/real/<name_lower>/`` in the same ``X_*.npy``/``Y_*.npy``/``meta.json``
shape ``causaltemp_xai.real_data.load_real_dataset`` expects -- but with
``"graph": null`` in ``meta.json`` rather than a placeholder graph (M4b has
none, by design).

Default target is BasicMotions (`DECISIONS.md` 2026-08-05): 6 channels, 4
classes, 40 train / 40 test, T=100 -- small, clean, matches this repo's
smoke-scale culture.

Usage
-----
    uv run python experiments/00_prepare_real_dataset.py --name BasicMotions
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.data_io import stratified_split  # noqa: E402
from causaltemp_xai.real_data import DEFAULT_REAL_DIR, parse_ts_file  # noqa: E402

UEA_BASE_URL = "https://www.timeseriesclassification.com/aeon-toolkit"


def download_and_extract(name: str, cache_dir: Path) -> Path:
    """Download ``<name>.zip`` from the UEA archive and extract it to ``cache_dir``.

    Skips the download if ``<name>_TRAIN.ts`` already exists in ``cache_dir``
    (idempotent -- re-running this script does not re-fetch the archive).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_ts = cache_dir / f"{name}_TRAIN.ts"
    if train_ts.exists():
        print(f"[00] {train_ts} already present, skipping download")
        return cache_dir

    url = f"{UEA_BASE_URL}/{name}.zip"
    zip_path = cache_dir / f"{name}.zip"
    print(f"[00] downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"[00] extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    return cache_dir


def prepare(name: str, out_dir: Path | str = DEFAULT_REAL_DIR, seed: int = 0) -> Path:
    """Download (if needed), parse, split, and cache ``name`` under ``out_dir``."""
    cache_dir = Path("data/real_cache") / name
    download_and_extract(name, cache_dir)

    X_train_full, y_train_full_str = parse_ts_file(cache_dir / f"{name}_TRAIN.ts")
    X_test, y_test_str = parse_ts_file(cache_dir / f"{name}_TEST.ts")

    class_names = sorted(set(y_train_full_str.tolist()) | set(y_test_str.tolist()))
    label_to_int = {c: i for i, c in enumerate(class_names)}
    y_train_full = np.array([label_to_int[c] for c in y_train_full_str])
    y_test = np.array([label_to_int[c] for c in y_test_str])

    # Archive TEST stays held out; split archive TRAIN into train/val only
    # (80/20, stratified) -- matches this repo's stratified_split helper,
    # reused rather than reimplemented.
    tr_idx, val_idx, _ = stratified_split(y_train_full, seed=seed, fractions=(0.8, 0.2, 0.0))

    dest = Path(out_dir) / name.lower()
    dest.mkdir(parents=True, exist_ok=True)

    np.save(dest / "X_train.npy", X_train_full[tr_idx])
    np.save(dest / "Y_train.npy", y_train_full[tr_idx])
    np.save(dest / "X_val.npy", X_train_full[val_idx])
    np.save(dest / "Y_val.npy", y_train_full[val_idx])
    np.save(dest / "X_test.npy", X_test)
    np.save(dest / "Y_test.npy", y_test)

    meta = {
        "name": name,
        "source": f"{UEA_BASE_URL}/{name}.zip",
        "graph": None,
        "class_names": class_names,
        "seed": seed,
        "shapes": {
            "X_train": list(X_train_full[tr_idx].shape),
            "X_val": list(X_train_full[val_idx].shape),
            "X_test": list(X_test.shape),
        },
        "split_sizes": {
            "train": len(tr_idx),
            "val": len(val_idx),
            "test": len(y_test),
        },
    }
    with open(dest / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[00] wrote {name} -> {dest}")
    print(f"     split sizes: {meta['split_sizes']}")
    print(f"     classes: {class_names}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="BasicMotions", help="UEA archive dataset name")
    parser.add_argument("--out-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.name, out_dir=args.out_dir, seed=args.seed)
