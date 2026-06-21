"""Tests for the NlinearSCMT (additive-noise MLP-mechanism) generator.

Mirrors ``tests/test_generator.py`` but targets the nonlinear generator. Verifies:

- Output dict shapes for X, Y, graph; mechanism is an ``MLPMechanism``.
- Binary, both-classes-present, ~balanced labels (median-threshold rule).
- **Boundedness / finiteness** over a long horizon across many seeds (the key
  stability requirement).
- Graph sparsity is close to the requested target (shared graph sampler).
- **Data-level nonlinearity**: a best linear VAR(L) fit leaves a clearly larger
  residual on NlinearSCMT data than on LinearSCMT data.
- Reproducibility under a fixed seed, *including* the divergence-resample path.
- KS test rejects the Gaussian hypothesis for the marginal.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from causaltemp_xai.benchmark.generator import LinearSCMT, NlinearSCMT
from causaltemp_xai.benchmark.mechanisms import MLPMechanism


# ---------------------------------------------------------------------------
# Output shape / structure
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_x_shape(self):
        gen = NlinearSCMT(k=4, L=2, T=30, N=50, seed=0)
        data = gen.generate()
        assert data["X"].shape == (50, 30, 4)

    def test_y_shape(self):
        gen = NlinearSCMT(k=4, L=2, T=30, N=50, seed=0)
        data = gen.generate()
        assert data["Y"].shape == (50,)

    def test_graph_shape(self):
        gen = NlinearSCMT(k=5, L=3, T=20, N=10, seed=1)
        data = gen.generate()
        assert data["graph"].shape == (5, 5, 3)

    def test_mechanism_is_mlp(self):
        gen = NlinearSCMT(k=5, L=3, T=20, N=10, seed=1)
        data = gen.generate()
        assert isinstance(data["mechanism"], MLPMechanism)
        assert data["mechanism"].L == 3 and data["mechanism"].k == 5

    def test_labels_binary(self):
        gen = NlinearSCMT(k=3, L=1, T=20, N=100, seed=3)
        data = gen.generate()
        assert set(data["Y"]).issubset({0, 1})

    def test_labels_both_classes_present(self):
        gen = NlinearSCMT(k=3, L=1, T=20, N=200, seed=3)
        data = gen.generate()
        assert 0 in data["Y"] and 1 in data["Y"]

    def test_labels_balanced(self):
        gen = NlinearSCMT(k=4, L=2, T=30, N=400, seed=5)
        data = gen.generate()
        frac = data["Y"].mean()
        # Median threshold ⇒ ~50/50 (ties may nudge it slightly off 0.5).
        assert 0.4 <= frac <= 0.6

    def test_reproducibility(self):
        gen_a = NlinearSCMT(k=3, L=2, T=20, N=30, seed=42)
        gen_b = NlinearSCMT(k=3, L=2, T=20, N=30, seed=42)
        np.testing.assert_array_equal(gen_a.generate()["X"], gen_b.generate()["X"])


# ---------------------------------------------------------------------------
# Boundedness / stability (the key requirement)
# ---------------------------------------------------------------------------


class TestStability:
    def test_finite_and_bounded_long_horizon(self):
        """Trajectories stay finite & bounded over T=200 across >=20 seeds."""
        for seed in range(20):
            gen = NlinearSCMT(k=6, L=2, T=200, N=200, seed=seed)
            X = gen.generate()["X"]
            assert np.all(np.isfinite(X)), f"non-finite at seed {seed}"
            assert np.max(np.abs(X)) <= gen.clip, f"unbounded at seed {seed}"

    def test_no_nan_or_inf(self):
        gen = NlinearSCMT(k=5, L=2, T=50, N=50, seed=7)
        data = gen.generate()
        assert np.all(np.isfinite(data["X"]))


# ---------------------------------------------------------------------------
# Graph sparsity (shared sampler ⇒ same guarantee as linear)
# ---------------------------------------------------------------------------


class TestGraphSparsity:
    @pytest.mark.parametrize("sparsity", [0.1, 0.3, 0.5, 0.7])
    def test_sparsity_within_tolerance(self, sparsity):
        gen = NlinearSCMT(k=8, L=2, sparsity=sparsity, T=10, N=10, seed=0)
        data = gen.generate()
        actual = data["graph"].mean()
        assert abs(actual - sparsity) < 0.15, (
            f"sparsity={sparsity}: actual={actual:.3f} out of tolerance"
        )


# ---------------------------------------------------------------------------
# Data-level nonlinearity
# ---------------------------------------------------------------------------


def _var_fit_residual(X: np.ndarray, L: int) -> float:
    """Normalised residual of the best least-squares linear VAR(L) fit to X.

    Builds the lagged design matrix (windows of the L previous steps) and the
    next-step targets pooled over all samples/timesteps, solves the linear map by
    least squares, and returns ``||residual|| / ||centred target||``. A linear SCM
    is (up to noise) explained by such a fit → small residual; a genuinely
    nonlinear SCM leaves a clearly larger residual.
    """
    N, T, k = X.shape
    feats, targets = [], []
    for t in range(L, T):
        # Flatten the L-step window per sample into a (k*L,) feature vector.
        win = X[:, t - L : t, :].reshape(N, L * k)
        feats.append(win)
        targets.append(X[:, t, :])
    F = np.concatenate(feats, axis=0)
    Y = np.concatenate(targets, axis=0)
    # Augment with an intercept column.
    F = np.hstack([F, np.ones((F.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(F, Y, rcond=None)
    resid = Y - F @ coef
    denom = np.linalg.norm(Y - Y.mean(axis=0)) + 1e-12
    return float(np.linalg.norm(resid) / denom)


class TestNonlinearity:
    def test_linear_fit_residual_larger_than_linear_scm(self):
        """A linear VAR fit explains LinearSCMT far better than NlinearSCMT."""
        k, L, T, N, seed = 5, 2, 60, 300, 11
        lin = LinearSCMT(k=k, L=L, T=T, N=N, seed=seed).generate()["X"]
        nlin = NlinearSCMT(k=k, L=L, T=T, N=N, seed=seed).generate()["X"]
        r_lin = _var_fit_residual(lin, L)
        r_nlin = _var_fit_residual(nlin, L)
        # Loose margin to avoid flakiness: nonlinear residual is clearly larger.
        assert r_nlin > r_lin * 1.2, (
            f"residual not clearly larger: linear={r_lin:.3f} nonlinear={r_nlin:.3f}"
        )


# ---------------------------------------------------------------------------
# Divergence-resample path (forced)
# ---------------------------------------------------------------------------


class TestDivergenceResample:
    def _diverging_gen(self, seed):
        """A generator whose mechanism is monkeypatched to blow up, with a clip
        small enough that the resample path fires."""
        gen = NlinearSCMT(k=3, L=2, T=20, N=40, seed=seed, clip=1.0, max_resample=3)
        # Replace with a high-gain, large-weight mechanism so tanh saturates near
        # +/-gain every step and the additive noise pushes |x| past clip=1.0.
        rng = np.random.default_rng(seed + 1000)
        gen.mechanism = MLPMechanism.random(
            gen.graph, hidden=8, rng=rng, gain=5.0, spectral_cap=10.0, init_gain=5.0
        )
        return gen

    def test_forced_divergence_output_is_finite(self):
        gen = self._diverging_gen(seed=0)
        X = gen.generate()["X"]
        # Resample never tames a deterministically-diverging mechanism, so the clip
        # fallback must guarantee a finite, bounded result.
        assert np.all(np.isfinite(X))
        assert np.max(np.abs(X)) <= gen.clip

    def test_forced_divergence_is_reproducible(self):
        a = self._diverging_gen(seed=0).generate()["X"]
        b = self._diverging_gen(seed=0).generate()["X"]
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Noise distribution
# ---------------------------------------------------------------------------


class TestNoiseDist:
    def test_laplace_rejects_gaussian(self):
        gen = NlinearSCMT(k=1, L=1, noise_type="laplace", T=500, N=200, seed=0)
        data = gen.generate()
        samples = data["X"].ravel()
        z = (samples - samples.mean()) / (samples.std() + 1e-9)
        _, p = kstest(z, "norm")
        assert p < 0.05, f"KS p-value {p:.4f} did not reject Gaussian"

    def test_invalid_noise_type_raises(self):
        with pytest.raises(ValueError):
            NlinearSCMT(k=3, L=1, noise_type="invalid", seed=0)
