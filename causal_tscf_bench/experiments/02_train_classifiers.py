"""
Script 02: Train TCN, LSTM, and Transformer classifiers on a benchmark dataset.

Usage:
  python experiments/02_train_classifiers.py --benchmark LinearSCMT
  python experiments/02_train_classifiers.py --benchmark NlinearSCMT --classifiers tcn lstm

Target: ≥ 90% validation accuracy before proceeding to CF evaluation.
"""

import argparse
import pathlib
import numpy as np

from causal_tscf_bench.benchmarks.linear_scm_t import LinearSCMT
from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Nonmonotonic, NlinearSCMT_Regime
from causal_tscf_bench.classifiers.tcn import TCNClassifier
from causal_tscf_bench.classifiers.lstm import LSTMClassifier
from causal_tscf_bench.classifiers.transformer import TransformerClassifier

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
TARGET_ACC = 0.90

BENCH_MAP = {
    "LinearSCMT": LinearSCMT,
    "NlinearSCMT": NlinearSCMT,
    "NlinearSCMT_Nonmonotonic": NlinearSCMT_Nonmonotonic,
    "NlinearSCMT_Regime": NlinearSCMT_Regime,
}

CLF_MAP = {
    "tcn": TCNClassifier,
    "lstm": LSTMClassifier,
    "transformer": TransformerClassifier,
}


def train_classifier(bench_name: str, clf_name: str) -> None:
    bclass = BENCH_MAP[bench_name]
    bench = bclass.__new__(bclass)
    split = bench.load(str(DATA_DIR / bench_name))

    clf = CLF_MAP[clf_name]()
    print(f"[02] Training {clf_name} on {bench_name} ...")
    result = clf.fit(
        split.X_train, split.Y_train,
        X_val=split.X_val, Y_val=split.Y_val,
    )
    val_acc = result["best_val_acc"]
    print(f"     Best val accuracy: {val_acc:.4f}")
    if val_acc < TARGET_ACC:
        print(f"     WARNING: val acc {val_acc:.3f} < target {TARGET_ACC:.2f}")

    out_dir = MODEL_DIR / bench_name / clf_name
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save(str(out_dir))
    print(f"     Saved -> {out_dir}")

    test_acc = float(np.mean(clf.predict(split.X_test) == split.Y_test))
    print(f"     Test accuracy: {test_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCH_MAP))
    parser.add_argument("--classifiers", nargs="+", default=["tcn", "lstm", "transformer"])
    args = parser.parse_args()

    for clf_name in args.classifiers:
        train_classifier(args.benchmark, clf_name)


if __name__ == "__main__":
    main()
