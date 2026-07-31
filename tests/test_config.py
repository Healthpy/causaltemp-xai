"""Tests for causaltemp_xai.config's ad hoc config derivation helpers.

``shifted_config`` already has coverage in tests/test_shift.py; this file
covers ``seeded_variant`` (M2, multi-seed protocol), which varies ``seed``
instead of ``noise_type`` -- and, unlike ``shifted_config``, seed changes the
SCM graph/mechanism too (they are sampled from the seed), not just the noise
marginal.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.benchmarks.generator import (
    LinearSCMT,
    NlinearSCMT,
    RegimeSwitchNlinearSCMT,
)
from causaltemp_xai.config import (
    CONFIGS,
    SMOKE,
    SMOKE_GAUSSIAN,
    SMOKE_NL,
    SMOKE_NONMONOTONIC,
    SMOKE_REGIME,
    seeded_variant,
)
from causaltemp_xai.data_io import build_generator


class TestSeededVariant:
    def test_changes_only_seed(self):
        variant = seeded_variant(SMOKE, seed=7)
        assert variant.seed == 7
        assert variant.name == "smoke_seed7"
        for field in ("k", "L", "sparsity", "noise_type", "T", "N", "mechanism_type", "nonlinear"):
            assert getattr(variant, field) == getattr(SMOKE, field)

    def test_distinct_name_even_when_seed_equals_base_seed(self):
        """Deliberately not aliased: reusing base.seed still gets its own
        on-disk name, per the docstring's "never silently aliases" contract."""
        variant = seeded_variant(SMOKE, seed=SMOKE.seed)
        assert variant.seed == SMOKE.seed
        assert variant.name != SMOKE.name
        assert variant.name == f"{SMOKE.name}_seed{SMOKE.seed}"

    def test_custom_name_override(self):
        variant = seeded_variant(SMOKE, seed=3, name="my_custom_name")
        assert variant.name == "my_custom_name"
        assert variant.seed == 3

    def test_deterministic_given_same_seed(self):
        v1 = seeded_variant(SMOKE, seed=5)
        v2 = seeded_variant(SMOKE, seed=5)
        assert v1 == v2  # frozen dataclass -> value equality

    def test_different_seeds_produce_different_scm(self):
        """Unlike shifted_config (noise-only), seeded_variant changes the seed
        that drives graph + mechanism sampling -- two different seeds should
        (with overwhelming probability, k=5 random sparsity-0.2 graphs) give a
        different graph and/or different mechanism coefficients."""
        v_a = seeded_variant(SMOKE, seed=1)
        v_b = seeded_variant(SMOKE, seed=2)

        gen_a = LinearSCMT(
            k=v_a.k,
            L=v_a.L,
            sparsity=v_a.sparsity,
            noise_type=v_a.noise_type,
            T=v_a.T,
            N=v_a.N,
            seed=v_a.seed,
        )
        gen_b = LinearSCMT(
            k=v_b.k,
            L=v_b.L,
            sparsity=v_b.sparsity,
            noise_type=v_b.noise_type,
            T=v_b.T,
            N=v_b.N,
            seed=v_b.seed,
        )

        graph_a, graph_b = gen_a.graph, gen_b.graph
        same_graph = np.array_equal(graph_a, graph_b)
        same_coeffs = same_graph and all(
            np.allclose(a, b) for a, b in zip(gen_a.mechanism.A_list, gen_b.mechanism.A_list)
        )
        assert not (same_graph and same_coeffs), (
            "two different seeds produced a bit-identical graph AND mechanism "
            "-- multi-seed replicates would not actually vary the SCM"
        )

    def test_same_seed_reproduces_identical_scm(self):
        """Sanity check the flip side: seeded_variant(SMOKE, seed=X) called
        twice must yield bit-identical generated data (determinism)."""
        v1 = seeded_variant(SMOKE, seed=9)
        v2 = seeded_variant(SMOKE, seed=9)
        gen1 = LinearSCMT(
            k=v1.k,
            L=v1.L,
            sparsity=v1.sparsity,
            noise_type=v1.noise_type,
            T=v1.T,
            N=v1.N,
            seed=v1.seed,
        )
        gen2 = LinearSCMT(
            k=v2.k,
            L=v2.L,
            sparsity=v2.sparsity,
            noise_type=v2.noise_type,
            T=v2.T,
            N=v2.N,
            seed=v2.seed,
        )
        np.testing.assert_array_equal(gen1.graph, gen2.graph)
        data1, data2 = gen1.generate(), gen2.generate()
        np.testing.assert_array_equal(data1["X"], data2["X"])
        np.testing.assert_array_equal(data1["Y"], data2["Y"])


# ---------------------------------------------------------------------------
# M4 benchmark-extension ablation presets (H5/H6/H7) -- SMOKE-SCALE ONLY.
# These tests check config-level correctness (registration, "vary exactly
# one field", reproducibility) of the three new presets; see
# tests/test_generator.py, tests/test_nlinear_generator.py, and
# tests/test_mechanisms.py for the generator/mechanism-level structural
# properties (non-monotonicity, regime-parameter difference, noise-shape
# distinguishability), and docs/archive/m4_ablation_presets_smoke.md for the design
# + smoke-scale preliminary findings. No full-scale variant of any of these
# three presets exists.
# ---------------------------------------------------------------------------


class TestSmokeGaussianPreset:
    """H5 negative control: SMOKE with Gaussian innovation noise."""

    def test_registered_and_varies_only_noise_type(self):
        assert CONFIGS["smoke_gaussian"] is SMOKE_GAUSSIAN
        assert SMOKE_GAUSSIAN.noise_type == "gaussian"
        assert SMOKE_GAUSSIAN.name == "smoke_gaussian"
        for field in ("k", "L", "sparsity", "T", "N", "seed", "mechanism_type", "nonlinear"):
            assert getattr(SMOKE_GAUSSIAN, field) == getattr(SMOKE, field)

    def test_build_generator_dispatches_linear_with_gaussian_noise(self):
        gen = build_generator(SMOKE_GAUSSIAN)
        assert isinstance(gen, LinearSCMT)
        assert gen.noise_type == "gaussian"

    def test_reproducible_given_fixed_seed(self):
        gen_a = build_generator(SMOKE_GAUSSIAN)
        gen_b = build_generator(SMOKE_GAUSSIAN)
        np.testing.assert_array_equal(gen_a.generate()["X"], gen_b.generate()["X"])


class TestSmokeNonmonotonicPreset:
    """H6 non-monotonic mechanism ablation: SMOKE_NL with a non-monotonic
    hidden activation instead of tanh."""

    def test_registered_and_varies_only_activation(self):
        assert CONFIGS["smoke_nonmonotonic"] is SMOKE_NONMONOTONIC
        assert SMOKE_NONMONOTONIC.mechanism_type == "mlp"
        assert SMOKE_NONMONOTONIC.nonlinear["activation"] == "nonmonotonic"
        for field in ("hidden", "gain", "decay_range", "spectral_cap", "init_gain"):
            assert SMOKE_NONMONOTONIC.nonlinear[field] == SMOKE_NL.nonlinear[field]
        for field in ("k", "L", "sparsity", "noise_type", "T", "N", "seed"):
            assert getattr(SMOKE_NONMONOTONIC, field) == getattr(SMOKE_NL, field)

    def test_build_generator_returns_nlinear_with_nonmonotonic_activation(self):
        gen = build_generator(SMOKE_NONMONOTONIC)
        assert isinstance(gen, NlinearSCMT)
        assert gen.mechanism.activation == "nonmonotonic"

    def test_reproducible_given_fixed_seed(self):
        gen_a = build_generator(SMOKE_NONMONOTONIC)
        gen_b = build_generator(SMOKE_NONMONOTONIC)
        np.testing.assert_array_equal(gen_a.generate()["X"], gen_b.generate()["X"])


class TestSmokeRegimePreset:
    """H7 regime-switching ablation: a single deterministic structural break
    at T/2, layered on the SMOKE_NL-scale nonlinear mechanism."""

    def test_registered_and_dispatches_to_regime_switch_generator(self):
        assert CONFIGS["smoke_regime"] is SMOKE_REGIME
        assert SMOKE_REGIME.mechanism_type == "mlp_regime_switch"
        gen = build_generator(SMOKE_REGIME)
        assert isinstance(gen, RegimeSwitchNlinearSCMT)

    def test_other_fields_match_smoke_nl(self):
        for field in ("k", "L", "sparsity", "noise_type", "T", "N", "seed"):
            assert getattr(SMOKE_REGIME, field) == getattr(SMOKE_NL, field)

    def test_two_regimes_have_deterministically_different_parameters(self):
        """Non-overlapping decay_range + distinct gain by construction --
        a cheaply checkable structural property, not just 'it ran'."""
        gen = build_generator(SMOKE_REGIME)
        assert gen.mechanism1.decay.min() > gen.mechanism2.decay.max()
        assert gen.mechanism1.gain != gen.mechanism2.gain

    def test_reproducible_given_fixed_seed(self):
        gen_a = build_generator(SMOKE_REGIME)
        gen_b = build_generator(SMOKE_REGIME)
        data_a, data_b = gen_a.generate(), gen_b.generate()
        np.testing.assert_array_equal(data_a["X"], data_b["X"])
        np.testing.assert_array_equal(data_a["Y"], data_b["Y"])
        assert data_a["switch_t"] == data_b["switch_t"]
