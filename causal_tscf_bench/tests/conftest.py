"""Shared pytest fixtures."""

import numpy as np
import pytest

from causal_tscf_bench.scm.dag import sample_dag
from causal_tscf_bench.scm.operators import sample_mechanism, INVERTIBLE_OPERATORS
from causal_tscf_bench.scm.tscm import simulate_tscm, sample_noise


@pytest.fixture(scope="session")
def small_dag():
    return sample_dag(n_channels=4, max_lag=2, edge_density=0.4,
                      rng=np.random.default_rng(42))


@pytest.fixture(scope="session")
def small_mechanisms(small_dag):
    return sample_mechanism(small_dag, rng=np.random.default_rng(43),
                            operator_pool=list(INVERTIBLE_OPERATORS))


@pytest.fixture(scope="session")
def small_X(small_dag, small_mechanisms):
    noise = sample_noise(N=50, T_total=30, M=4, distribution="laplace",
                         rng=np.random.default_rng(44))
    return simulate_tscm(small_dag, small_mechanisms, noise)
