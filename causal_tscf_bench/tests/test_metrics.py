"""Tests for Axis A-D metric functions."""

import numpy as np
import pytest

from causal_tscf_bench.metrics.axis_a import icc, mcc_concept, latent_disentanglement
from causal_tscf_bench.metrics.axis_b import shd, lag_accuracy, graph_auc
from causal_tscf_bench.metrics.axis_c import (
    validity, proximity, sparsity, trsi, cf_faith, ivr
)
from causal_tscf_bench.metrics.axis_d import input_sensitivity


# ---- Axis A ----

def test_icc_perfect():
    phi = np.zeros((10, 4))
    phi[:, 2] = 1.0
    assert icc(phi, int_channel=2) == pytest.approx(1.0)


def test_icc_zero():
    phi = np.zeros((10, 4))
    assert icc(phi, int_channel=2) == pytest.approx(0.0)


def test_mcc_concept_full_coverage():
    phi = np.ones((10, 4))
    assert mcc_concept(phi, [0, 1, 2, 3]) == pytest.approx(1.0)


def test_mcc_concept_empty_parents():
    phi = np.ones((10, 4))
    result = mcc_concept(phi, [])
    assert np.isnan(result)


def test_latent_disentanglement_perfect():
    N, latent_dim, M = 200, 4, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, M))
    Z = X + rng.normal(0, 0.01, (N, latent_dim))
    ld = latent_disentanglement(Z, X)
    assert ld > 0.9


# ---- Axis B ----

def test_shd_perfect():
    adj = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    assert shd(adj, adj) == 0


def test_shd_all_wrong():
    adj_true = np.ones((2, 2, 2), dtype=int)
    adj_pred = np.zeros((2, 2, 2), dtype=int)
    assert shd(adj_true, adj_pred) == 8


def test_lag_accuracy_perfect():
    adj = np.zeros((3, 3, 2), dtype=int)
    adj[0, 1, 0] = 1
    adj[2, 1, 1] = 1
    assert lag_accuracy(adj, adj) == pytest.approx(1.0)


def test_graph_auc_all_correct():
    adj = np.array([[0, 1], [0, 0]])
    score = np.array([[0.1, 0.9], [0.1, 0.1]])
    auc = graph_auc(adj, score)
    assert auc == pytest.approx(1.0)


# ---- Axis C ----

def test_proximity_identical():
    X = np.random.randn(5, 20, 3)
    assert proximity(X, X.copy(), normalize=False) == pytest.approx(0.0, abs=1e-6)


def test_sparsity_identical():
    X = np.random.randn(5, 20, 3)
    assert sparsity(X, X.copy()) == pytest.approx(1.0)


def test_sparsity_all_changed():
    X = np.zeros((5, 20, 3))
    X_cf = np.ones((5, 20, 3))
    assert sparsity(X, X_cf) == pytest.approx(0.0)


def test_trsi_constant_perturbation():
    X = np.zeros((5, 20, 3))
    X_cf = np.ones((5, 20, 3))   # constant perturbation → zero first difference
    assert trsi(X_cf, X) == pytest.approx(0.0, abs=1e-10)


def test_trsi_noisy_perturbation():
    rng = np.random.default_rng(0)
    X = np.zeros((5, 20, 3))
    X_cf = rng.normal(0, 1, (5, 20, 3))
    assert trsi(X_cf, X) > 0.0


def test_cf_faith_perfect():
    x = np.random.randn(20, 3)
    # X'_exp = X'_CF → DTW(X'_exp, X'_CF) = 0 → CF-faith = 1
    assert cf_faith(x, x, x * 2) == pytest.approx(1.0)


def test_ivr_no_violations():
    X = np.zeros((5, 20, 3))
    X_cf = X.copy()
    X_cf[:, 15:, :] = 1.0   # only change after T_int=15
    assert ivr(X_cf, X, T_int=15) == pytest.approx(0.0)


def test_ivr_all_violations():
    X = np.zeros((5, 20, 3))
    X_cf = np.ones((5, 20, 3))   # change before T_int too
    assert ivr(X_cf, X, T_int=15) == pytest.approx(1.0)


# ---- Axis D ----

def test_input_sensitivity_stable():
    """A constant-output attribution should have near-zero sensitivity."""
    X = np.random.randn(3, 10, 2)
    const_attr = lambda x: np.zeros_like(x)
    sens = input_sensitivity(X, const_attr, eps=0.1)
    assert sens == pytest.approx(0.0, abs=1e-8)
