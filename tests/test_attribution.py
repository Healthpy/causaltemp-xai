"""Tests for the Integrated-Gradients foil, deletion/insertion curves, and the
two disclosed-proxy attribution baselines (FDSaliency, MCMaskSHAP).

See causaltemp_xai/methods/attribution/fd_saliency.py and mc_mask_shap.py for
the full disclosure of what these two classes do and do not implement
relative to Dynamask (Crabbe & van der Schaar, 2021) and TimeSHAP (Bento et
al., 2021), and docs/method_provenance.md for the provenance summary.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causaltemp_xai.methods.attribution import (
    deletion_curve,
    insertion_curve,
    integrated_gradients,
    FDSaliency,
    MCMaskSHAP,
)
from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier


@pytest.fixture(scope="module")
def trained():
    gen = LinearSCMT(k=5, L=1, T=20, N=240, seed=0)
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    clf = LSTMClassifier(
        n_inputs=5,
        hidden_size=16,
        num_layers=2,
        dropout=0.0,
        lr=5e-3,
        batch_size=64,
        max_epochs=25,
        patience=25,
        seed=0,
    )
    clf.fit(X[:200], Y[:200], X[200:], Y[200:])
    return clf, data


def _target_logit(clf, arr, target_class):
    """Scalar target-class logit f_target(arr) via the differentiable hook."""
    with torch.no_grad():
        logits = clf.torch_logits(torch.as_tensor(np.asarray(arr, dtype=np.float32)))
    return float(logits[0, target_class].item())


class TestIntegratedGradients:
    def test_shape(self, trained):
        clf, data = trained
        x = data["X"][0]
        ig = integrated_gradients(clf, x, target_class=1, steps=32)
        assert ig.shape == x.shape
        assert np.all(np.isfinite(ig))

    def test_completeness_axiom(self, trained):
        """sum(IG) ≈ f_target(x) − f_target(baseline) (the IG completeness axiom)."""
        clf, data = trained
        x = data["X"][0]
        baseline = np.zeros_like(x)
        ig = integrated_gradients(clf, x, target_class=1, baseline=baseline, steps=256)

        delta_f = _target_logit(clf, x, 1) - _target_logit(clf, baseline, 1)
        assert ig.sum() == pytest.approx(delta_f, abs=1e-2)

    def test_custom_baseline_shape_checked(self, trained):
        clf, data = trained
        x = data["X"][0]
        with pytest.raises(ValueError):
            integrated_gradients(clf, x, target_class=1, baseline=np.zeros((3, 3)))


class TestPerturbationCurves:
    def test_curves_length_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        ig = integrated_gradients(clf, x, target_class=1, steps=32)

        dele, dele_auc = deletion_curve(clf, x, ig, target_class=1, n_steps=20)
        ins, ins_auc = insertion_curve(clf, x, ig, target_class=1, n_steps=20)

        assert dele.shape == (21,) and ins.shape == (21,)
        assert np.all(np.isfinite(dele)) and np.all(np.isfinite(ins))
        assert np.isfinite(dele_auc) and np.isfinite(ins_auc)

    def test_endpoints_are_sensible(self, trained):
        clf, data = trained
        x = data["X"][0]
        baseline = np.zeros_like(x)
        ig = integrated_gradients(clf, x, target_class=1, steps=32)

        full_conf = float(clf.predict_proba(x[np.newaxis])[0, 1])
        base_conf = float(clf.predict_proba(baseline[np.newaxis])[0, 1])

        dele, _ = deletion_curve(clf, x, ig, target_class=1, n_steps=20)
        ins, _ = insertion_curve(clf, x, ig, target_class=1, n_steps=20)

        # Deletion starts at full confidence and ends at the baseline.
        assert dele[0] == pytest.approx(full_conf, abs=1e-5)
        assert dele[-1] == pytest.approx(base_conf, abs=1e-5)
        # Insertion starts at the baseline and ends at full confidence.
        assert ins[0] == pytest.approx(base_conf, abs=1e-5)
        assert ins[-1] == pytest.approx(full_conf, abs=1e-5)


class TestFDSaliency:
    """FDSaliency: finite-difference saliency (the honestly-renamed proxy that
    used to ship as ``Dynamask``). See fd_saliency.py for the full disclosure
    -- this is a numerical-gradient baseline, NOT Dynamask's learned soft
    mask.
    """

    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        phi = FDSaliency().attribute(x, clf, target_class=1)
        assert phi.shape == x.shape
        assert np.all(np.isfinite(phi))

    def test_default_target_class_matches_predicted_class(self, trained):
        """target_class=None falls back to the classifier's own prediction."""
        clf, data = trained
        x = data["X"][0]
        pred = int(clf.predict(x[np.newaxis])[0])
        phi_default = FDSaliency().attribute(x, clf, target_class=None)
        phi_explicit = FDSaliency().attribute(x, clf, target_class=pred)
        assert np.array_equal(phi_default, phi_explicit)

    def test_eps_is_the_only_effective_hyperparameter(self, trained):
        """Two different eps values should (generically) yield different
        finite-difference magnitudes -- eps is the only knob this class
        exposes that affects the computed attribution.
        """
        clf, data = trained
        x = data["X"][0]
        phi_small = FDSaliency(eps=1e-3).attribute(x, clf, target_class=1)
        phi_large = FDSaliency(eps=1e-1).attribute(x, clf, target_class=1)
        assert not np.array_equal(phi_small, phi_large)


class TestMCMaskSHAP:
    """MCMaskSHAP: Monte-Carlo random-coalition Shapley approximation (the
    honestly-renamed, now-seeded proxy that used to ship as ``TimeSHAP`` with
    an unseeded RNG). See mc_mask_shap.py for the full disclosure -- this is
    a flat random-coalition sampler, NOT TimeSHAP's pruning/hierarchy scheme.
    """

    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        phi = MCMaskSHAP(n_samples=30, seed=0).attribute(x, clf, target_class=1)
        assert phi.shape == x.shape
        assert np.all(np.isfinite(phi))

    def test_same_seed_is_bit_identical(self, trained):
        """Same seed, same inputs -> bit-identical output across two calls
        (the reproducibility bug in the old unseeded TimeSHAP-named class is
        fixed by the explicit ``seed`` constructor argument).
        """
        clf, data = trained
        x = data["X"][0]
        phi_a = MCMaskSHAP(n_samples=50, seed=7).attribute(x, clf, target_class=1)
        phi_b = MCMaskSHAP(n_samples=50, seed=7).attribute(x, clf, target_class=1)
        assert np.array_equal(phi_a, phi_b)

    def test_different_seeds_generally_differ(self, trained):
        clf, data = trained
        x = data["X"][0]
        phi_a = MCMaskSHAP(n_samples=50, seed=1).attribute(x, clf, target_class=1)
        phi_b = MCMaskSHAP(n_samples=50, seed=2).attribute(x, clf, target_class=1)
        assert not np.array_equal(phi_a, phi_b)

    def test_seed_none_defaults_to_fixed_reproducible_seed(self, trained):
        """``seed=None`` documents "use the fixed codebase default (0)", not
        "be nondeterministic" -- two fresh instances with no seed argument
        must therefore also agree bit-for-bit (see MCMaskSHAP's docstring).
        """
        clf, data = trained
        x = data["X"][0]
        phi_a = MCMaskSHAP(n_samples=30).attribute(x, clf, target_class=1)
        phi_b = MCMaskSHAP(n_samples=30).attribute(x, clf, target_class=1)
        assert np.array_equal(phi_a, phi_b)
        # And it must match passing the default seed explicitly.
        phi_explicit = MCMaskSHAP(n_samples=30, seed=0).attribute(x, clf, target_class=1)
        assert np.array_equal(phi_a, phi_explicit)
