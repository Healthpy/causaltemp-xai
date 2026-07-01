"""
Script 03: Run CF explainers and attribution methods on test instances.

Saves X_cf_exp (N, T, M) and attributions (N, T, M) for each method × benchmark.

Usage:
  python experiments/03_run_methods.py --benchmark LinearSCMT --classifier tcn --methods comte tsevo
"""

import argparse
import pathlib
import numpy as np
import pickle
import torch
import torch.nn as nn

from causal_tscf_bench.benchmarks.linear_scm_t import LinearSCMT
from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Nonmonotonic, NlinearSCMT_Regime
from causal_tscf_bench.classifiers.tcn import TCNClassifier
from causal_tscf_bench.classifiers.lstm import LSTMClassifier
from causal_tscf_bench.classifiers.transformer import TransformerClassifier
from causal_tscf_bench.methods.counterfactual.wachter import WachterCF
from causal_tscf_bench.methods.counterfactual.comte import ComteCF
from causal_tscf_bench.methods.counterfactual.tsevo import TSEvoCF
from causal_tscf_bench.methods.counterfactual.glacier import GlacierCF
from causal_tscf_bench.methods.counterfactual.cels import CELSCF
from causal_tscf_bench.methods.counterfactual.confetti import ConfettiCF
from causal_tscf_bench.methods.attribution.timeshap import TimeSHAP
from causal_tscf_bench.methods.attribution.dynamask import Dynamask
from causal_tscf_bench.methods.causal.carla_causal import CARLACausal

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

BENCH_MAP = {
    "LinearSCMT": LinearSCMT,
    "NlinearSCMT": NlinearSCMT,
    "NlinearSCMT_Nonmonotonic": NlinearSCMT_Nonmonotonic,
    "NlinearSCMT_Regime": NlinearSCMT_Regime,
}
CLF_MAP = {"tcn": TCNClassifier, "lstm": LSTMClassifier, "transformer": TransformerClassifier}
CF_METHODS = {
    "wachter": WachterCF,
    "comte": ComteCF,
    "tsevo": TSEvoCF,
    "glacier": GlacierCF,
    "cels": CELSCF,
    "confetti": ConfettiCF,
    "carla": CARLACausal,
}
ATTR_METHODS = {"timeshap": TimeSHAP, "dynamask": Dynamask}


class _ModelShapeAdapter(nn.Module):
    """Normalize cfts's varied input shapes to (N, M, T) for our classifiers.

    cfts methods call classifier.model directly with shapes that may differ from
    our (N, M, T) convention:
      - (N, T, M)   — tsevo and others that don't auto-transpose
      - (N, 1, M*T) — comte, wachter flattened paths

    This adapter detects the orientation and corrects it so the LSTM always
    receives (N, M, T) regardless of what cfts passes.
    """
    def __init__(self, model: nn.Module, n_channels: int, seq_len: int):
        super().__init__()
        self._m = model
        self._C = n_channels  # M
        self._L = seq_len     # T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            N, d1, d2 = x.shape
            if d1 == 1 and d2 == self._C * self._L:
                # Flattened (N, 1, M*T) → (N, M, T)
                x = x.reshape(N, self._C, self._L)
            elif d1 == self._L and d2 == self._C:
                # Transposed (N, T, M) → (N, M, T)
                x = x.permute(0, 2, 1)
            # else: already (N, M, T) or some other shape — pass through
        return self._m(x)


def run_cf_method(method_name: str, split, classifier, out_dir: pathlib.Path) -> np.ndarray:
    # Wrap classifier.model so cfts calls receive shape-corrected tensors
    M = split.X_train.shape[2]
    T = split.X_train.shape[1]
    original_model = classifier.model
    classifier.model = _ModelShapeAdapter(original_model, n_channels=M, seq_len=T)

    try:
        explainer = CF_METHODS[method_name]()
        explainer.fit(split.X_train, classifier)

        n_test = len(split.X_test)
        X_cf = np.zeros_like(split.X_test)
        target_class = 1 - int(split.Y_test[0])   # flip from majority

        for i in range(n_test):
            X_cf[i] = explainer.explain(split.X_test[i], target_class, classifier)
            if (i + 1) % 50 == 0:
                print(f"     [{i+1}/{n_test}]")

        np.save(str(out_dir / f"X_cf_{method_name}.npy"), X_cf)
    finally:
        classifier.model = original_model  # always restore

    return X_cf


def run_attr_method(method_name: str, split, classifier, out_dir: pathlib.Path) -> np.ndarray:
    method = ATTR_METHODS[method_name]()
    if hasattr(method, "fit"):
        method.fit(split.X_train)

    n_test = len(split.X_test)
    attrs = np.zeros_like(split.X_test)
    for i in range(n_test):
        attrs[i] = method.attribute(split.X_test[i], classifier)
    np.save(str(out_dir / f"attr_{method_name}.npy"), attrs)
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCH_MAP))
    parser.add_argument("--classifier", required=True, choices=list(CLF_MAP))
    parser.add_argument("--methods", nargs="+", default=list(CF_METHODS))
    parser.add_argument("--n_test", type=int, default=None, help="Limit test instances (default: all)")
    args = parser.parse_args()

    bench_cls = BENCH_MAP[args.benchmark]
    bench = bench_cls.__new__(bench_cls)
    split = bench.load(str(DATA_DIR / args.benchmark))

    # Optionally cap the test set for faster iteration
    if args.n_test is not None:
        from causal_tscf_bench.benchmarks.base import BenchmarkSplit
        split = BenchmarkSplit(
            X_train=split.X_train, Y_train=split.Y_train,
            X_val=split.X_val, Y_val=split.Y_val,
            X_test=split.X_test[:args.n_test], Y_test=split.Y_test[:args.n_test],
            dag=split.dag, mechanisms=split.mechanisms, meta=split.meta,
        )
        print(f"[03] Using {args.n_test} test instances")

    clf = CLF_MAP[args.classifier]()
    clf.load(str(MODEL_DIR / args.benchmark / args.classifier))

    out_dir = RESULTS_DIR / args.benchmark / args.classifier
    out_dir.mkdir(parents=True, exist_ok=True)

    for method_name in args.methods:
        print(f"[03] Running {method_name} on {args.benchmark} / {args.classifier} ...")
        if method_name in CF_METHODS:
            run_cf_method(method_name, split, clf, out_dir)
        elif method_name in ATTR_METHODS:
            run_attr_method(method_name, split, clf, out_dir)
        else:
            print(f"     Unknown method: {method_name}")

    print("[03] Done.")


if __name__ == "__main__":
    main()
