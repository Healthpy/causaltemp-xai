"""Tests for ChannelConceptProbe (formerly ``CBMT``; R3 remediation).

See docs/method_provenance.md for the naming-provenance note: this is a
per-channel logistic probe over summary statistics, not a concept bottleneck
model. These tests wire the public export to a test per R9 (a public export
with no test and no experiment wiring is a standing-rule violation) and check
the probe's basic contract: per-channel fit, correct output shape, and a
time-uniform map.
"""

from __future__ import annotations

import numpy as np

from causaltemp_xai.methods import ChannelConceptProbe
from causaltemp_xai.methods.concept.channel_concept_probe import ChannelConceptProbe as _Direct


def _make_toy_dataset(n=60, T=15, k=3, seed=0):
    rng = np.random.default_rng(seed)
    # Channel 0's mean is the (noisy) discriminative signal for concept 0.
    concept0 = rng.integers(0, 2, size=n)
    X = rng.normal(scale=0.1, size=(n, T, k))
    X[:, :, 0] += concept0[:, None] * 2.0  # shifts the mean for concept0=1
    concept_labels = np.zeros((n, k), dtype=int)
    concept_labels[:, 0] = concept0
    # Other channels: random labels uncorrelated with the data (probe should
    # still fit without error, just less informatively).
    concept_labels[:, 1:] = rng.integers(0, 2, size=(n, k - 1))
    return X, concept_labels


def test_public_export_is_the_module_class():
    """causaltemp_xai.methods.ChannelConceptProbe is the concept-module class."""
    assert ChannelConceptProbe is _Direct


def test_fit_concepts_trains_one_probe_per_channel():
    X, labels = _make_toy_dataset()
    probe = ChannelConceptProbe()
    probe.fit_concepts(X, labels)
    assert probe._n_concepts == X.shape[2]
    assert len(probe._probes) == X.shape[2]


def test_attribute_shape_and_time_uniformity():
    X, labels = _make_toy_dataset()
    probe = ChannelConceptProbe()
    probe.fit_concepts(X, labels)

    x = X[0]
    phi = probe.attribute(x, classifier=None)
    assert phi.shape == x.shape
    # Time-uniform: every row equal (disclosed behaviour, not a bug).
    for m in range(x.shape[1]):
        assert np.allclose(phi[:, m], phi[0, m])


def test_attribute_before_fit_returns_zeros():
    probe = ChannelConceptProbe()
    x = np.zeros((10, 3))
    phi = probe.attribute(x, classifier=None)
    assert phi.shape == (10, 3)
    assert np.all(phi == 0.0)


def test_probe_separates_informative_concept():
    """Channel 0's probe should recover P(concept=1) > 0.5 for a high-mean
    instance and < 0.5 for a low-mean instance, given the clearly separable
    toy construction."""
    X, labels = _make_toy_dataset(n=200, seed=1)
    probe = ChannelConceptProbe()
    probe.fit_concepts(X, labels)

    x_high = X[labels[:, 0] == 1][0]
    x_low = X[labels[:, 0] == 0][0]
    phi_high = probe.attribute(x_high, classifier=None)
    phi_low = probe.attribute(x_low, classifier=None)
    assert phi_high[0, 0] > phi_low[0, 0]
