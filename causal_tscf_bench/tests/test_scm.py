"""Tests for SCM core: DAG, operators, simulate, abduction, counterfactual."""

import numpy as np
import pytest

from causal_tscf_bench.scm.dag import sample_dag
from causal_tscf_bench.scm.operators import sample_mechanism, INVERTIBLE_OPERATORS
from causal_tscf_bench.scm.tscm import simulate_tscm, sample_noise
from causal_tscf_bench.scm.abduction import abduct
from causal_tscf_bench.scm.counterfactual import compute_gt_counterfactual


@pytest.fixture
def simple_dag():
    rng = np.random.default_rng(0)
    return sample_dag(n_channels=4, max_lag=2, edge_density=0.4, rng=rng)


@pytest.fixture
def mechanisms(simple_dag):
    rng = np.random.default_rng(1)
    return sample_mechanism(simple_dag, rng=rng, operator_pool=list(INVERTIBLE_OPERATORS))


def test_dag_has_at_least_one_edge(simple_dag):
    assert simple_dag.adjacency.sum() >= 1


def test_dag_shape(simple_dag):
    assert simple_dag.adjacency.shape == (4, 4, 2)


def test_simulate_returns_correct_shape(simple_dag, mechanisms):
    N, T, M = 10, 30, 4
    rng = np.random.default_rng(2)
    noise = sample_noise(N=N, T_total=T, M=M, distribution="laplace", rng=rng)
    X = simulate_tscm(simple_dag, mechanisms, noise, burn_in=5)
    assert X.shape == (N, T, M)


def test_simulate_no_nans(simple_dag, mechanisms):
    N, T, M = 5, 20, 4
    rng = np.random.default_rng(3)
    noise = sample_noise(N, T, M, "laplace", rng)
    X = simulate_tscm(simple_dag, mechanisms, noise)
    assert not np.isnan(X).any()


def test_abduction_round_trip(simple_dag, mechanisms):
    """Abduction should exactly recover noise from simulated data."""
    N, T, M = 5, 30, 4
    rng = np.random.default_rng(4)
    noise_true = sample_noise(N, T, M, "laplace", rng)
    X = simulate_tscm(simple_dag, mechanisms, noise_true)
    U_recovered = abduct(X, simple_dag, mechanisms)
    max_lag = simple_dag.max_lag
    # Check only post-burn-in steps (t >= max_lag)
    assert U_recovered.shape == noise_true.shape
    np.testing.assert_allclose(
        U_recovered[:, max_lag:], noise_true[:, max_lag:], atol=1e-6
    )


def test_counterfactual_shape(simple_dag, mechanisms):
    N, T, M = 3, 30, 4
    rng = np.random.default_rng(5)
    noise = sample_noise(N, T, M, "laplace", rng)
    X = simulate_tscm(simple_dag, mechanisms, noise)
    X_cf = compute_gt_counterfactual(
        X[0], simple_dag, mechanisms,
        int_channel=0, int_time=15, int_value=1.0
    )
    assert X_cf.shape == (T, M)


def test_counterfactual_pre_intervention_unchanged(simple_dag, mechanisms):
    N, T, M = 2, 30, 4
    rng = np.random.default_rng(6)
    noise = sample_noise(N, T, M, "laplace", rng)
    X = simulate_tscm(simple_dag, mechanisms, noise)
    T_int = 15
    X_cf = compute_gt_counterfactual(
        X[0], simple_dag, mechanisms, int_channel=0, int_time=T_int, int_value=2.0
    )
    # All time steps before T_int must equal factual
    np.testing.assert_allclose(X_cf[:T_int], X[0, :T_int], atol=1e-6)
