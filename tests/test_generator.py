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
from scipy.stats import kstest, kurtosis

from causaltemp_xai.benchmarks.generator import (
    _GAUSSIAN_STD,
    LinearSCMT,
    NlinearSCMT,
    exogenous_channels,
)
from causaltemp_xai.config import CONFIGS, FULL, FULL_NL, SMOKE
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
        assert (
            abs(actual - sparsity) < 0.15
        ), f"sparsity={sparsity}: actual={actual:.3f} out of tolerance"


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

    # -----------------------------------------------------------------
    # M4/H5 negative-control ablation: Gaussian innovation noise.
    # -----------------------------------------------------------------

    def test_gaussian_does_not_reject_gaussian(self):
        """Sanity-check the flip side of the two tests above: a KS test
        should *not* reject the Gaussian hypothesis for Gaussian noise."""
        gen = LinearSCMT(k=1, L=1, noise_type="gaussian", T=500, N=200, seed=0)
        data = gen.generate()
        samples = data["X"].ravel()
        z = (samples - samples.mean()) / (samples.std() + 1e-9)
        _, p = kstest(z, "norm")
        assert p > 0.05, f"KS p-value {p:.4f} rejected Gaussian for Gaussian noise"

    def test_gaussian_noise_variance_matches_laplace_variance(self):
        """H5 isolates distribution *shape*, not scale: the Gaussian std is
        chosen so its variance matches Laplace(scale=0.1)'s variance
        (2*scale**2=0.02) closely -- confirms the ablation varies noise
        shape, not innovation scale."""
        rng = np.random.default_rng(1)
        lap = rng.laplace(loc=0.0, scale=0.1, size=500_000)
        gauss = rng.normal(loc=0.0, scale=_GAUSSIAN_STD, size=500_000)
        assert abs(lap.var() - gauss.var()) < 0.001

    def test_gaussian_distinguishable_from_laplace_by_kurtosis(self):
        """Cheaply checkable structural property (not just 'it ran'):
        Laplace is leptokurtic (excess kurtosis = 3 analytically) at any
        scale; Gaussian's excess kurtosis is 0 by definition -- the two
        innovation distributions are shape-distinguishable even at matched
        variance."""
        rng = np.random.default_rng(2)
        lap = rng.laplace(loc=0.0, scale=0.1, size=200_000)
        gauss = rng.normal(loc=0.0, scale=_GAUSSIAN_STD, size=200_000)
        k_lap = kurtosis(lap, fisher=True)
        k_gauss = kurtosis(gauss, fisher=True)
        assert k_lap > 1.5, f"Laplace excess kurtosis too low: {k_lap:.2f}"
        assert abs(k_gauss) < 0.5, f"Gaussian excess kurtosis should be ~0: {k_gauss:.2f}"

    def test_gaussian_generate_is_reproducible(self):
        gen_a = LinearSCMT(k=3, L=1, noise_type="gaussian", T=20, N=50, seed=5)
        gen_b = LinearSCMT(k=3, L=1, noise_type="gaussian", T=20, N=50, seed=5)
        np.testing.assert_array_equal(gen_a.generate()["X"], gen_b.generate()["X"])


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


class TestExogenousChannels:
    """M3's U_s/U_d/V typing prerequisite (`TSCausalCF`, was `CausalFeasibilityCF`).

    Decided 2026-08-04: a channel with zero incoming edges is
    this benchmark's ``U_d`` (dynamic exogenous); every other channel is ``V``.
    ``U_s`` has no counterpart here and is always empty. The decision was made
    *after* checking real data, not assumed -- these numbers pin that check so
    a future graph/sparsity change surfaces here rather than silently changing
    the typing everyone downstream assumes.
    """

    def test_no_incoming_edges_means_pure_noise(self):
        """A channel with an all-zero row has nothing to be exogenous *from*."""
        graph = np.zeros((4, 4, 1))
        graph[1, 0, 0] = 1  # channel 1 <- channel 0
        graph[2, 1, 0] = 1  # channel 2 <- channel 1
        # channels 0 and 3 have no incoming edge at all.
        assert exogenous_channels(graph) == [0, 3]

    def test_fully_connected_graph_has_no_exogenous_channels(self):
        graph = np.ones((3, 3, 1))
        np.fill_diagonal(graph[:, :, 0], 0.0)  # lag-1 self-loops never occur
        assert exogenous_channels(graph) == []

    def test_empty_graph_is_all_exogenous(self):
        graph = np.zeros((5, 5, 2))
        assert exogenous_channels(graph) == [0, 1, 2, 3, 4]

    def test_full_and_full_nl_have_no_exogenous_channels_at_paper_scale(self):
        """The finding that shaped the M3 decision: U_d is empty where the
        DoD needs it. Sparsity 0.2 at k=10 leaves every channel a descendant
        of at least one other -- not a bug to route around."""
        for cfg, cls in ((FULL, LinearSCMT), (FULL_NL, NlinearSCMT)):
            scm = cls(k=cfg.k, L=cfg.L, sparsity=cfg.sparsity, seed=cfg.seed)
            assert exogenous_channels(scm.graph) == []

    def test_smoke_has_one_exogenous_channel(self):
        """The one config where the typing is non-degenerate."""
        scm = LinearSCMT(k=SMOKE.k, L=SMOKE.L, sparsity=SMOKE.sparsity, seed=SMOKE.seed)
        assert exogenous_channels(scm.graph) == [1]


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
