"""Tests for CITRIS (B4) — data support + genuine-upstream-model plumbing.

These verify the *interfaces and training mechanics* (intervention-labeled data
generation, that the model uses the genuine vendored upstream CITRIS modules,
CITRIS-VAE ELBO training, the Axis-B graph interface shapes). They deliberately
do NOT assert graph-recovery quality: CITRIS is a heavy VAE whose disentangle-
ment needs proper-scale training (see the ``CITRIS`` class docstring), so a
quality assertion at the tiny scale used here would be flaky rather than a guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.generator import NlinearSCMT
from causaltemp_xai.benchmarks.interventional import (
    InterventionalDataset,
    generate_interventional_sequences,
)
from causaltemp_xai.methods.causal import CITRIS


@pytest.fixture(scope="module")
def scm():
    return NlinearSCMT(k=4, L=1, sparsity=0.3, T=20, N=80, seed=0)


class TestInterventionalDataSupport:
    def test_shapes_and_dtypes(self, scm):
        ds = generate_interventional_sequences(
            scm.mechanism, k=4, L=1, T=20, N=80, seed=1, intervention_prob=0.2
        )
        assert isinstance(ds, InterventionalDataset)
        assert ds.X.shape == (80, 20, 4)
        assert ds.targets.shape == (80, 20, 4)
        assert ds.values.shape == (80, 20, 4)
        assert set(np.unique(ds.targets)).issubset({0, 1})

    def test_first_step_never_intervened(self, scm):
        ds = generate_interventional_sequences(
            scm.mechanism, k=4, L=1, T=20, N=80, seed=1, intervention_prob=0.5
        )
        assert ds.targets[:, 0, :].sum() == 0

    def test_intervention_fraction_matches_prob(self, scm):
        ds = generate_interventional_sequences(
            scm.mechanism, k=4, L=1, T=20, N=200, seed=2, intervention_prob=0.3
        )
        # Only steps t>=1 can be intervened, so the overall fraction is
        # slightly below the per-eligible-step prob.
        frac = ds.targets.mean()
        assert 0.2 < frac < 0.32

    def test_do_values_recorded_where_intervened(self, scm):
        ds = generate_interventional_sequences(
            scm.mechanism, k=4, L=1, T=20, N=80, seed=3, intervention_prob=0.3
        )
        mask = ds.targets == 1
        # Where intervened, X equals the recorded do-value exactly.
        np.testing.assert_allclose(ds.X[mask], ds.values[mask])

    def test_reproducible(self, scm):
        a = generate_interventional_sequences(scm.mechanism, k=4, L=1, T=20, N=50, seed=5)
        b = generate_interventional_sequences(scm.mechanism, k=4, L=1, T=20, N=50, seed=5)
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.targets, b.targets)

    def test_invalid_prob_raises(self, scm):
        with pytest.raises(ValueError):
            generate_interventional_sequences(
                scm.mechanism, k=4, L=1, T=20, N=10, intervention_prob=1.5
            )


class TestCITRIS:
    @pytest.fixture(scope="class")
    def fitted(self, scm):
        ds = generate_interventional_sequences(
            scm.mechanism, k=4, L=1, T=20, N=80, seed=1,
            intervention_prob=0.4, mode="single",
        )
        model = CITRIS(
            k=4, latents_per_block=2, c_hid=16, max_epochs=4, seed=0
        ).fit(ds.X, ds.targets)
        return model, ds

    def test_uses_upstream_transition_prior(self, fitted):
        """The identifiability-critical prior must be the genuine vendored one."""
        model, _ = fitted
        assert type(model.model.prior).__module__ == "models.shared.transition_prior"
        assert type(model.model.intv_classifier).__module__ == "models.shared.target_classifier"

    def test_training_reduces_loss(self, fitted):
        model, _ = fitted
        assert len(model.history_) == 4
        assert model.history_[-1] < model.history_[0]

    def test_encode_shape(self, fitted):
        model, ds = fitted
        z = model.encode(ds.X[:5])
        # num_latents = k * latents_per_block
        assert z.shape == (5, 20, model.model.num_latents)
        assert model.model.num_latents == 4 * 2

    def test_inferred_graph_shapes_and_range(self, fitted):
        model, _ = fitted
        adj, scores = model.inferred_graph(max_lag=1, threshold=0.2)
        assert adj.shape == (4, 4, 1)
        assert scores.shape == (4, 4, 1)
        assert adj.dtype == int
        assert set(np.unique(adj)).issubset({0, 1})
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        # No self-loops (diagonal zeroed to match ground-truth convention).
        assert np.all(np.diagonal(scores[:, :, 0]) == 0.0)

    def test_wrong_channel_count_raises(self, scm):
        ds = generate_interventional_sequences(scm.mechanism, k=4, L=1, T=20, N=20, seed=1)
        with pytest.raises(ValueError):
            CITRIS(k=5, max_epochs=1).fit(ds.X, ds.targets)

    def test_inferred_graph_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CITRIS(k=4).inferred_graph()
