"""Tests for benchmark dataset generation."""

import numpy as np
import pytest

from causal_tscf_bench.benchmarks.linear_scm_t import LinearSCMT
from causal_tscf_bench.benchmarks.nlinear_scm_t import NlinearSCMT
from causal_tscf_bench.benchmarks.nlinear_ablations import NlinearSCMT_Nonmonotonic, NlinearSCMT_Regime


def make_linear(seed=0):
    return LinearSCMT(
        n_samples=300, T=30, M=4, max_lag=2, edge_density=0.4,
        noise_distribution="laplace", noise_scale=0.1,
        rng=np.random.default_rng(seed),
    )


def make_nlinear(seed=0):
    return NlinearSCMT(
        n_samples=300, T=30, M=4, max_lag=2, edge_density=0.4,
        noise_distribution="laplace", noise_scale=0.1,
        rng=np.random.default_rng(seed),
    )


def test_linear_generate_shape():
    split = make_linear().generate()
    assert split.X_train.ndim == 3
    assert split.X_train.shape[2] == 4


def test_linear_label_balance():
    split = make_linear().generate()
    balance = split.Y_train.mean()
    assert 0.3 <= balance <= 0.7, f"Balance {balance:.3f} out of [0.3, 0.7]"


def test_nlinear_no_nans():
    split = make_nlinear().generate()
    for X in [split.X_train, split.X_val, split.X_test]:
        assert not np.isnan(X).any()


def test_nlinear_label_balance():
    split = make_nlinear().generate()
    balance = split.Y_train.mean()
    assert 0.3 <= balance <= 0.7


def test_nonmonotonic_meta_flag():
    bench = NlinearSCMT_Nonmonotonic(
        n_samples=200, T=20, M=4, max_lag=2, edge_density=0.3,
        noise_distribution="laplace", noise_scale=0.1,
        rng=np.random.default_rng(0),
    )
    split = bench.generate()
    assert split.meta.get("identifiable") is False


def test_regime_switching_shape():
    bench = NlinearSCMT_Regime(
        n_samples=200, T=30, M=4, max_lag=2, edge_density=0.3,
        noise_distribution="laplace", noise_scale=0.1,
        n_regimes=2, min_regime_len=5,
        rng=np.random.default_rng(0),
    )
    split = bench.generate()
    assert split.X_train.shape[1] == 30


def test_split_sizes():
    split = make_linear(seed=7).generate()
    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total == 300
