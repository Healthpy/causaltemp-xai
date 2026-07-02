"""
Abstract base class for all TSCF benchmarks.

Adapted from causal_tscf_bench/causal_tscf_bench/benchmarks/base.py with these
changes:
  - Split ratio: 60/20/20 (train/val/test) instead of bench's 70/15/15
  - Mechanism field stores causaltemp_xai's Mechanism (LinearMechanism/MLPMechanism)
    object rather than bench's list[Mechanism] dataclasses, for backward compat
  - Uses causaltemp_xai's existing data_io.stratified_split() for splitting
  - BenchmarkSplit stores graph as ndarray AND optionally a LaggedDAG

Every benchmark ships five artefacts:
  X          (N, T, k)     — observed multivariate time series
  Y          (N,)           — binary class label
  graph      (k, k, L)     — ground-truth lagged adjacency (numpy)
  mechanism               — causaltemp_xai Mechanism object (LinearMechanism/MLPMechanism)
  meta       dict          — config and provenance metadata
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# causaltemp_xai split ratio (locked per data_io.py SPLIT_FRACTIONS)
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


@dataclass
class BenchmarkSplit:
    """Container for train/val/test splits plus SCM artefacts.

    Compatible with causaltemp_xai's data_io.py conventions.
    The `mechanism` field stores a causaltemp_xai Mechanism object
    (LinearMechanism or MLPMechanism from benchmarks/mechanisms.py).
    """
    X_train: np.ndarray       # (N_train, T, k)
    Y_train: np.ndarray       # (N_train,)
    X_val: np.ndarray         # (N_val, T, k)
    Y_val: np.ndarray         # (N_val,)
    X_test: np.ndarray        # (N_test, T, k)
    Y_test: np.ndarray        # (N_test,)
    graph: np.ndarray         # (k, k, L) binary adjacency
    mechanism: Any            # causaltemp_xai Mechanism (LinearMechanism or MLPMechanism)
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        """Number of variables/channels."""
        return self.X_train.shape[2]

    @property
    def T(self) -> int:
        """Time series length."""
        return self.X_train.shape[1]

    @property
    def L(self) -> int:
        """Maximum lag (from graph shape)."""
        return self.graph.shape[2]


class BenchmarkDataset(ABC):
    """Abstract base for all benchmark datasets.

    Adapted from causal_tscf_bench/benchmarks/base.py.
    Uses 60/20/20 split ratio (causaltemp_xai standard).
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self._split: BenchmarkSplit | None = None

    @abstractmethod
    def generate(self) -> BenchmarkSplit:
        """Generate (or regenerate) the dataset from config and seed."""
        ...

    def get_split(self) -> BenchmarkSplit:
        if self._split is None:
            self._split = self.generate()
        return self._split

    def save(self, out_dir: str | Path) -> None:
        """Save all split artefacts to out_dir/."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        split = self.get_split()
        np.save(out / "X_train.npy", split.X_train)
        np.save(out / "Y_train.npy", split.Y_train)
        np.save(out / "X_val.npy", split.X_val)
        np.save(out / "Y_val.npy", split.Y_val)
        np.save(out / "X_test.npy", split.X_test)
        np.save(out / "Y_test.npy", split.Y_test)
        np.save(out / "graph.npy", split.graph)
        np.savez(out / "mechanism.npz", **split.mechanism.state_dict())
        (out / "meta.json").write_text(json.dumps(split.meta, indent=2))

    @classmethod
    def load(cls, in_dir: str | Path) -> BenchmarkSplit:
        """Load a previously saved BenchmarkSplit from in_dir/."""
        from causaltemp_xai.benchmarks.mechanisms import mechanism_from_state_dict

        d = Path(in_dir)
        with np.load(d / "mechanism.npz") as mf:
            mechanism = mechanism_from_state_dict(dict(mf))
        meta = json.loads((d / "meta.json").read_text())
        return BenchmarkSplit(
            X_train=np.load(d / "X_train.npy"),
            Y_train=np.load(d / "Y_train.npy"),
            X_val=np.load(d / "X_val.npy"),
            Y_val=np.load(d / "Y_val.npy"),
            X_test=np.load(d / "X_test.npy"),
            Y_test=np.load(d / "Y_test.npy"),
            graph=np.load(d / "graph.npy"),
            mechanism=mechanism,
            meta=meta,
        )

    @staticmethod
    def _stratified_split(
        Y: np.ndarray,
        seed: int,
        fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return disjoint train/val/test index arrays, stratified on Y.

        Matches data_io.stratified_split() exactly (60/20/20).
        """
        rng = np.random.default_rng(seed)
        train_idx: list[np.ndarray] = []
        val_idx: list[np.ndarray] = []
        test_idx: list[np.ndarray] = []

        for cls_label in np.unique(Y):
            idx = np.where(Y == cls_label)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(round(n * fractions[0]))
            n_val = int(round(n * fractions[1]))
            train_idx.append(idx[:n_train])
            val_idx.append(idx[n_train: n_train + n_val])
            test_idx.append(idx[n_train + n_val:])

        train = np.concatenate(train_idx)
        val = np.concatenate(val_idx)
        test = np.concatenate(test_idx)

        for split in (train, val, test):
            rng.shuffle(split)

        return train, val, test

    @staticmethod
    def _label_from_threshold(
        X: np.ndarray,
        causal_channel: int,
        theta: float | None = None,
        balance_tol: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """
        Label Visibility principle: Y = 1[X_T^(j*) > theta].
        Theta is auto-tuned if None so that class balance is in [0.3, 0.7].
        """
        final_vals = X[:, -1, causal_channel]
        if theta is None:
            theta = float(np.median(final_vals))
        Y = (final_vals > theta).astype(int)
        return Y, theta
