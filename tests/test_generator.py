"""Tests for LinearSCMT data generator.

Verifies:
- Output dict has correct shapes for X, Y, graph, mechanisms.
- Graph sparsity is close to the requested sparsity parameter.
- KS test rejects the Gaussian hypothesis for Laplace noise.
- Locked configs are acyclic across time (no contemporaneous edges).
- Dataset persistence: stratified 60/20/20 splits, both classes present,
  and a lossless generate_and_save -> load_dataset round trip.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.config import CONFIGS, SMOKE
from causaltemp_xai.data_io import (
    SPLIT_FRACTIONS,
    generate_and_save,
    load_dataset,
    stratified_split,
)


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
        assert data["mechanism"].L == 3

    def test_mechanisms_matrix_shapes(self):
        k, L = 4, 2
        gen = LinearSCMT(k=k, L=L, T=20, N=10, seed=2)
        data = gen.generate()
        for i, A in enumerate(data["mechanism"].A_list):
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


# ---------------------------------------------------------------------------
# Acyclicity of the locked configs
# ---------------------------------------------------------------------------


def _scm_from_config(cfg):
    """Construct a LinearSCMT for a BenchmarkConfig (no generate() call)."""
    return LinearSCMT(
        k=cfg.k,
        L=cfg.L,
        sparsity=cfg.sparsity,
        noise_type=cfg.noise_type,
        T=cfg.T,
        N=cfg.N,
        seed=cfg.seed,
    )


class TestAcyclicity:
    @pytest.mark.parametrize("name", sorted(CONFIGS))
    def test_no_contemporaneous_edges(self, name):
        """Locked configs have no instantaneous (lag-0) dependence.

        The graph tensor stores only lagged edges (shape ``(k, k, L)`` with all
        L slices at lags >= 1), so the VAR is acyclic across time by
        construction. With ``L=1`` the lag-1 matrix additionally carries no
        self-loop (zero diagonal), so there is no instantaneous self-dependence.
        """
        cfg = CONFIGS[name]
        scm = _scm_from_config(cfg)
        graph = scm.graph
        assert graph.shape == (cfg.k, cfg.k, cfg.L)
        # No self-loop at the first lag (generator zeros the lag-1 diagonal).
        assert np.all(np.diag(graph[:, :, 0]) == 0.0)


# ---------------------------------------------------------------------------
# Dataset persistence (Stage 2)
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    def test_sizes_are_60_20_20_and_disjoint(self):
        rng = np.random.default_rng(0)
        Y = rng.integers(0, 2, size=500)
        train, val, test = stratified_split(Y, seed=0)

        n = len(Y)
        assert len(train) + len(val) + len(test) == n
        # Disjoint coverage of all indices.
        all_idx = np.concatenate([train, val, test])
        assert len(np.unique(all_idx)) == n
        # Fractions within a tight tolerance of 60/20/20.
        assert abs(len(train) / n - SPLIT_FRACTIONS[0]) < 0.03
        assert abs(len(val) / n - SPLIT_FRACTIONS[1]) < 0.03
        assert abs(len(test) / n - SPLIT_FRACTIONS[2]) < 0.03

    def test_both_classes_in_every_split(self):
        rng = np.random.default_rng(1)
        Y = rng.integers(0, 2, size=500)
        train, val, test = stratified_split(Y, seed=0)
        for split in (train, val, test):
            classes = set(Y[split].tolist())
            assert classes == {0, 1}


class TestPersistenceRoundTrip:
    def test_save_creates_all_artifacts(self, tmp_path):
        dest = generate_and_save(SMOKE, out_dir=tmp_path)
        for split in ("train", "val", "test"):
            assert (dest / f"X_{split}.npy").exists()
            assert (dest / f"Y_{split}.npy").exists()
        assert (dest / "graph.npy").exists()
        assert (dest / "mechanism.npz").exists()
        assert (dest / "meta.json").exists()

    def test_meta_reports_splits_and_balance(self, tmp_path):
        generate_and_save(SMOKE, out_dir=tmp_path)
        loaded = load_dataset("smoke", out_dir=tmp_path)
        meta = loaded["meta"]
        sizes = meta["split_sizes"]
        assert sizes["train"] + sizes["val"] + sizes["test"] == SMOKE.N
        for split in ("train", "val", "test"):
            assert set(meta["class_balance"][split].keys()) == {"0", "1"}

    def test_round_trip_reproduces_arrays(self, tmp_path):
        """Two independent save/load cycles produce identical arrays."""
        generate_and_save(SMOKE, out_dir=tmp_path / "a")
        generate_and_save(SMOKE, out_dir=tmp_path / "b")
        a = load_dataset("smoke", out_dir=tmp_path / "a")
        b = load_dataset("smoke", out_dir=tmp_path / "b")

        for split in ("train", "val", "test"):
            assert np.array_equal(a[f"X_{split}"], b[f"X_{split}"])
            assert np.array_equal(a[f"Y_{split}"], b[f"Y_{split}"])
        assert np.array_equal(a["graph"], b["graph"])
        assert a["mechanism"].L == SMOKE.L == b["mechanism"].L
        for A_a, A_b in zip(a["mechanism"].A_list, b["mechanism"].A_list):
            assert np.array_equal(A_a, A_b)

    def test_load_matches_in_memory_generation(self, tmp_path):
        """Loaded splits match an in-memory generate + same stratified split."""
        generate_and_save(SMOKE, out_dir=tmp_path)
        loaded = load_dataset("smoke", out_dir=tmp_path)

        scm = _scm_from_config(SMOKE)
        data = scm.generate()
        train, val, test = stratified_split(data["Y"], seed=SMOKE.seed)
        expected = {"train": train, "val": val, "test": test}
        for split, idx in expected.items():
            assert np.array_equal(loaded[f"X_{split}"], data["X"][idx])
            assert np.array_equal(loaded[f"Y_{split}"], data["Y"][idx])
        assert np.array_equal(loaded["graph"], data["graph"])

