"""Tests for LinearSCMT data generator.

Verifies:
- Output dict has correct shapes for X, Y, graph, mechanisms.
- Graph sparsity is close to the requested sparsity parameter.
- KS test rejects the Gaussian hypothesis for Laplace noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from causaltemp_xai.benchmark.generator import LinearSCMT


# ---------------------------------------------------------------------------
# Output shape / structure
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_x_shape(self):
        gen = LinearSCMT(k=4, L=2, T=30, N=50, seed=0)
        data = gen.generate()
        assert data["X"].shape == (50, 30, 4)

    def test_y_shape(self):
        gen = LinearSCMT(k=4, L=2, T=30, N=50, seed=0)
        data = gen.generate()
        assert data["Y"].shape == (50,)

    def test_graph_shape(self):
        gen = LinearSCMT(k=5, L=3, T=20, N=10, seed=1)
        data = gen.generate()
        assert data["graph"].shape == (5, 5, 3)

    def test_mechanisms_length(self):
        gen = LinearSCMT(k=5, L=3, T=20, N=10, seed=1)
        data = gen.generate()
        assert len(data["mechanisms"]) == 3

    def test_mechanisms_matrix_shapes(self):
        k, L = 4, 2
        gen = LinearSCMT(k=k, L=L, T=20, N=10, seed=2)
        data = gen.generate()
        for i, A in enumerate(data["mechanisms"]):
            assert A.shape == (k, k), f"mechanism[{i}] has shape {A.shape}"

    def test_labels_binary(self):
        gen = LinearSCMT(k=3, L=1, T=20, N=100, seed=3)
        data = gen.generate()
        assert set(data["Y"]).issubset({0, 1})

    def test_labels_both_classes_present(self):
        gen = LinearSCMT(k=3, L=1, T=20, N=200, seed=3)
        data = gen.generate()
        assert 0 in data["Y"] and 1 in data["Y"]

    def test_reproducibility(self):
        gen_a = LinearSCMT(k=3, L=1, T=10, N=20, seed=42)
        gen_b = LinearSCMT(k=3, L=1, T=10, N=20, seed=42)
        np.testing.assert_array_equal(gen_a.generate()["X"], gen_b.generate()["X"])

    def test_no_nan_or_inf(self):
        gen = LinearSCMT(k=5, L=2, T=50, N=50, seed=7)
        data = gen.generate()
        assert np.all(np.isfinite(data["X"]))


# ---------------------------------------------------------------------------
# Graph sparsity
# ---------------------------------------------------------------------------


class TestGraphSparsity:
    @pytest.mark.parametrize("sparsity", [0.1, 0.3, 0.5, 0.7])
    def test_sparsity_within_tolerance(self, sparsity):
        """Actual edge density should be within 0.15 of the target sparsity."""
        gen = LinearSCMT(k=8, L=2, sparsity=sparsity, T=10, N=10, seed=0)
        data = gen.generate()
        actual = data["graph"].mean()
        assert abs(actual - sparsity) < 0.15, (
            f"sparsity={sparsity}: actual={actual:.3f} out of tolerance"
        )


# ---------------------------------------------------------------------------
# Noise distribution
# ---------------------------------------------------------------------------


class TestNoiseDist:
    def test_laplace_rejects_gaussian(self):
        """KS test should reject the Gaussian hypothesis for Laplace noise."""
        gen = LinearSCMT(k=1, L=1, noise_type="laplace", T=500, N=200, seed=0)
        data = gen.generate()
        # Flatten all time-series values as noise proxy (large sample)
        samples = data["X"].ravel()
        z = (samples - samples.mean()) / (samples.std() + 1e-9)
        _, p = kstest(z, "norm")
        assert p < 0.05, f"KS p-value {p:.4f} did not reject Gaussian for Laplace noise"

    def test_uniform_rejects_gaussian(self):
        """KS test should reject the Gaussian hypothesis for uniform noise."""
        gen = LinearSCMT(k=1, L=1, noise_type="uniform", T=500, N=200, seed=0)
        data = gen.generate()
        samples = data["X"].ravel()
        z = (samples - samples.mean()) / (samples.std() + 1e-9)
        _, p = kstest(z, "norm")
        assert p < 0.05, f"KS p-value {p:.4f} did not reject Gaussian for uniform noise"

    def test_invalid_noise_type_raises(self):
        with pytest.raises((ValueError, NotImplementedError)):
            LinearSCMT(k=3, L=1, noise_type="invalid", seed=0).generate()

