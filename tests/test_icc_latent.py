"""Tests for the corrected latent-traversal ICC (`axis_a.icc_latent`).

These prove the metric is trustworthy *independently of any particular VAE*,
using a synthetic encoder/decoder/classifier where exactly one latent dimension
causally controls the label. A correct ICC must then score that dimension high
and the others ~0 — and must do so via the matched reconstruction baseline
(Correction 1), so a lossy decoder does not inflate the non-causal dims.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.metrics.axis_a import icc_latent


class _World:
    """Synthetic latent world: z in R^d; the label is 1[z_0 > 0].

    encode(X) recovers z from X (X carries z in its first d features);
    decode(z) renders z back into X-space; classifier thresholds channel 0.
    Dim 0 is the ONLY causal concept; a large +/-delta on it flips the label,
    perturbing any other dim does nothing.
    """

    def __init__(self, N=400, d=5, T=4, recon_noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.T = d, d  # embed z into first d of a length-d "series"
        self.recon_noise = recon_noise
        self.rng = rng
        self.Z = rng.normal(size=(N, d))
        self.X = self.decode(self.Z)

    def encode(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 0, : self.d]  # first timestep holds z

    def decode(self, Z):
        Z = np.asarray(Z, dtype=float)
        N, d = Z.shape
        X = np.zeros((N, self.T, d))
        X[:, 0, :] = Z
        if self.recon_noise:
            X = X + self.rng.normal(scale=self.recon_noise, size=X.shape)
        return X

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, 0, 0] > 0).astype(int)  # label = 1[z_0 > 0]


def test_icc_high_for_causal_dim_low_for_others():
    w = _World(seed=1)
    icc = icc_latent(w.X, w.encode, w.decode, w, delta=2.0, scale_by_std=True, symmetric=True)
    assert icc.shape == (w.d,)
    # Dim 0 controls the label -> high ICC; others -> ~0.
    assert icc[0] > 0.25
    assert icc[1:].max() < 0.05
    assert icc[0] > icc[1:].max() + 0.2


def test_matched_baseline_absorbs_reconstruction_noise():
    """With a lossy decoder, the matched f(D(z)) baseline must keep non-causal
    dims near zero (Correction 1) — an f(x) baseline would inflate them."""
    w = _World(recon_noise=0.05, seed=2)
    icc = icc_latent(w.X, w.encode, w.decode, w, delta=2.0)
    # Non-causal dims stay low despite decoder noise (baseline is matched).
    assert icc[1:].mean() < 0.15
    assert icc[0] > icc[1:].mean()


def test_symmetric_averaging_and_shape():
    w = _World(seed=3)
    one_sided = icc_latent(w.X, w.encode, w.decode, w, delta=2.0, symmetric=False)
    two_sided = icc_latent(w.X, w.encode, w.decode, w, delta=2.0, symmetric=True)
    assert one_sided.shape == two_sided.shape == (w.d,)
    # Both flag dim 0 as the causal one.
    assert np.argmax(one_sided) == 0 and np.argmax(two_sided) == 0


def test_std_scaling_handles_heterogeneous_latent_scales():
    """A dim with tiny variance must not be spuriously ranked by a raw delta;
    std-scaling makes the causal dim win regardless of per-dim scale."""
    w = _World(seed=4)
    w.Z[:, 1] *= 0.01  # dim 1 near-constant, but non-causal
    w.X = w.decode(w.Z)
    icc = icc_latent(w.X, w.encode, w.decode, w, delta=2.0, scale_by_std=True)
    assert np.argmax(icc) == 0
