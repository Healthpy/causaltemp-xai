"""
Abstract base class for all TSCF benchmarks.

Every benchmark ships four artefacts:
  X          (N, T, M)     — observed multivariate time series
  Y          (N,)           — binary class label (Label Visibility principle)
  graph      (M, M, L)     — ground-truth lagged adjacency
  mechanisms  list[Mechanism] — per-edge weights and operators
  cf_index    dict           — mapping instance idx -> pre-computed X'_CF

The get_gt_counterfactual() method computes X'_CF on-the-fly using the
three-step causal ladder (Plan Phase 3) if it is not pre-cached.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..scm.dag import LaggedDAG
from ..scm.operators import Mechanism
from ..scm.counterfactual import compute_gt_counterfactual


@dataclass
class BenchmarkSplit:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    dag: LaggedDAG
    mechanisms: list[Mechanism]
    meta: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return self.X_train.shape[2]

    @property
    def T(self) -> int:
        return self.X_train.shape[1]


class BenchmarkDataset(ABC):
    """Abstract base for all benchmark datasets."""

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
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        split = self.get_split()
        np.save(out / "X_train.npy", split.X_train)
        np.save(out / "Y_train.npy", split.Y_train)
        np.save(out / "X_val.npy", split.X_val)
        np.save(out / "Y_val.npy", split.Y_val)
        np.save(out / "X_test.npy", split.X_test)
        np.save(out / "Y_test.npy", split.Y_test)
        np.save(out / "graph.npy", split.dag.adjacency)
        # Save mechanisms as structured array
        mech_data = {
            "channel_from": [m.channel_from for m in split.mechanisms],
            "channel_to": [m.channel_to for m in split.mechanisms],
            "lag": [m.lag for m in split.mechanisms],
            "weight": [m.weight for m in split.mechanisms],
            "operator_key": [m.operator_key for m in split.mechanisms],
        }
        np.savez(out / "mechanisms.npz", **mech_data)
        (out / "meta.json").write_text(json.dumps(split.meta, indent=2))

    @classmethod
    def load(cls, in_dir: str | Path) -> BenchmarkSplit:
        d = Path(in_dir)
        adj = np.load(d / "graph.npy")
        mech_npz = np.load(d / "mechanisms.npz", allow_pickle=True)
        mechanisms = [
            Mechanism(
                channel_from=int(mech_npz["channel_from"][k]),
                channel_to=int(mech_npz["channel_to"][k]),
                lag=int(mech_npz["lag"][k]),
                weight=float(mech_npz["weight"][k]),
                operator_key=str(mech_npz["operator_key"][k]),
            )
            for k in range(len(mech_npz["channel_from"]))
        ]
        dag = LaggedDAG(
            n_channels=adj.shape[0],
            max_lag=adj.shape[2],
            adjacency=adj,
        )
        meta = json.loads((d / "meta.json").read_text())
        return BenchmarkSplit(
            X_train=np.load(d / "X_train.npy"),
            Y_train=np.load(d / "Y_train.npy"),
            X_val=np.load(d / "X_val.npy"),
            Y_val=np.load(d / "Y_val.npy"),
            X_test=np.load(d / "X_test.npy"),
            Y_test=np.load(d / "Y_test.npy"),
            dag=dag,
            mechanisms=mechanisms,
            meta=meta,
        )

    def get_gt_counterfactual(
        self,
        X: np.ndarray,
        int_channel: int,
        int_time: int,
        int_value: float,
    ) -> np.ndarray:
        """
        Compute the analytical ground-truth counterfactual on demand.

        Uses the three-step causal ladder (Plan Phase 3): abduction ->
        action -> prediction.
        """
        split = self.get_split()
        return compute_gt_counterfactual(
            X=X,
            dag=split.dag,
            mechanisms=split.mechanisms,
            int_channel=int_channel,
            int_time=int_time,
            int_value=int_value,
        )

    @staticmethod
    def _label_from_threshold(
        X: np.ndarray,
        causal_channel: int,
        theta: float | None = None,
        balance_tol: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """
        Label Visibility principle (Plan Phase 2):
          Y = 1[X_T^(j*) > theta]
        Theta is auto-tuned if None so that class balance is in [0.3, 0.7].
        """
        final_vals = X[:, -1, causal_channel]
        if theta is None:
            # Bisect to find theta giving balance in [0.3+tol, 0.7-tol]
            lo, hi = float(np.percentile(final_vals, 5)), float(np.percentile(final_vals, 95))
            for _ in range(50):
                mid = (lo + hi) / 2.0
                balance = float((final_vals > mid).mean())
                if balance > 0.5 + balance_tol:
                    lo = mid
                elif balance < 0.5 - balance_tol:
                    hi = mid
                else:
                    break
            theta = mid
        Y = (final_vals > theta).astype(np.int64)
        return Y, theta
