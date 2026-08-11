"""Tests for M4c's empirical contraction-rate diagnostic (R9 wiring)."""

from __future__ import annotations

from causaltemp_xai.benchmarks.diagnostics import empirical_contraction_rate
from causaltemp_xai.benchmarks.generator import NlinearSCMT, SpringSCMT


class TestEmpiricalContractionRate:
    def test_shape_and_keys(self):
        gen = NlinearSCMT(k=4, L=1, T=10, N=2, seed=0, hidden=8)
        out = empirical_contraction_rate(gen.mechanism, T=20, n_pairs=5, seed=0)
        assert set(out) == {"rate", "ratio_curve", "n_pairs", "T"}
        assert out["ratio_curve"].shape == (21,)
        assert out["ratio_curve"][0] == 1.0  # by construction at t=0

    def test_reproducible_given_same_seed(self):
        gen = NlinearSCMT(k=4, L=1, T=10, N=2, seed=0, hidden=8)
        out1 = empirical_contraction_rate(gen.mechanism, T=20, n_pairs=5, seed=3)
        out2 = empirical_contraction_rate(gen.mechanism, T=20, n_pairs=5, seed=3)
        assert out1["rate"] == out2["rate"]

    def test_known_dissipative_mechanism_contracts_more_than_new_families(self):
        """MLPMechanism is contractive by design (decay + spectral-capped
        tanh branch) -- its measured rate must be clearly more negative than
        the M4c non-dissipative family, which is the whole point of this
        diagnostic (M4c DoD: measure, don't assume rho ~= 1)."""
        gen_mlp = NlinearSCMT(k=5, L=1, T=10, N=2, seed=0, hidden=16)
        gen_spring = SpringSCMT(n_particles=5, T=10, N=2, seed=0)

        r_mlp = empirical_contraction_rate(gen_mlp.mechanism, T=50, n_pairs=30, seed=1)
        r_spring = empirical_contraction_rate(gen_spring.mechanism, T=50, n_pairs=30, seed=1)

        assert r_mlp["rate"] < -0.1, "known-dissipative MLP should contract clearly"
        assert r_spring["rate"] > r_mlp["rate"]
