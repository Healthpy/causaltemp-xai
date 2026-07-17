"""Tests for the Integrated-Gradients foil, deletion/insertion curves, and the
two official attribution methods (TimeSHAP, Dynamask).

``TimeSHAP`` wraps the official ``timeshap`` library (Bento et al., 2021) and
``Dynamask`` wraps the authors' official implementation vendored at
third_party/dynamask_repo (Crabbe & van der Schaar, 2021). Each test class is
skipped when its backing dependency/submodule is unavailable. See
docs/method_provenance.md for the provenance summary.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.methods.attribution import (
    Dynamask,
    TimeSHAP,
    deletion_curve,
    insertion_curve,
    integrated_gradients,
)


def _module_available(name: str) -> bool:
    import importlib

    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# Importing the classes above always works (both wrappers import their heavy
# backends lazily inside attribute()/fit()); these flags gate only the classes
# whose backend/submodule is actually needed at run time.
_HAS_TIMESHAP = _module_available("timeshap")
_HAS_DYNAMASK = _module_available("attribution.mask")


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


@pytest.mark.skipif(
    not _HAS_DYNAMASK,
    reason="Dynamask submodule (third_party/dynamask_repo) not checked out",
)
class TestDynamask:
    """Dynamask: the official implementation (Crabbe & van der Schaar, 2021),
    vendored at third_party/dynamask_repo and wrapped to emit its learned soft
    mask as the ``(T, k)`` map. See
    causaltemp_xai/methods/attribution/dynamask.py for the adapter.
    """

    def test_shape_and_mask_in_unit_interval(self, trained):
        clf, data = trained
        x = data["X"][0]
        phi = Dynamask(n_epoch=50, seed=0).attribute(x, clf, target_class=1)
        assert phi.shape == x.shape
        assert np.all(np.isfinite(phi))
        # Dynamask returns a soft mask in [0, 1].
        assert phi.min() >= 0.0 and phi.max() <= 1.0

    def test_same_seed_is_reproducible(self, trained):
        """Same seed + inputs -> identical learned mask across two calls."""
        clf, data = trained
        x = data["X"][0]
        a = Dynamask(n_epoch=50, seed=7).attribute(x, clf, target_class=1)
        b = Dynamask(n_epoch=50, seed=7).attribute(x, clf, target_class=1)
        assert np.array_equal(a, b)

    def test_class_agnostic_target(self, trained):
        """Dynamask preserves the whole predictive distribution, so the mask is
        class-agnostic: passing a different target_class does not change it.
        """
        clf, data = trained
        x = data["X"][0]
        phi_a = Dynamask(n_epoch=50, seed=0).attribute(x, clf, target_class=0)
        phi_b = Dynamask(n_epoch=50, seed=0).attribute(x, clf, target_class=1)
        assert np.array_equal(phi_a, phi_b)


@pytest.mark.skipif(not _HAS_TIMESHAP, reason="official timeshap not installed")
class TestTimeSHAP:
    """TimeSHAP: the official feedzai implementation (Bento et al., 2021),
    wrapped to emit a dense ``(T, k)`` map. See
    causaltemp_xai/methods/attribution/timeshap.py for the adapter design
    (pruning bypassed, all cells forced, event -> timestep pivot).
    """

    def test_shape_and_finite(self, trained):
        clf, data = trained
        x = data["X"][0]
        expl = TimeSHAP(nsamples=100, seed=0)
        expl.fit(data["X"][:200])
        phi = expl.attribute(x, clf, target_class=1)
        assert phi.shape == x.shape
        assert np.all(np.isfinite(phi))

    def test_same_seed_is_reproducible(self, trained):
        """Same seed + inputs -> identical output across two calls."""
        clf, data = trained
        x = data["X"][0]
        a = TimeSHAP(nsamples=100, seed=7)
        a.fit(data["X"][:200])
        b = TimeSHAP(nsamples=100, seed=7)
        b.fit(data["X"][:200])
        assert np.array_equal(
            a.attribute(x, clf, target_class=1), b.attribute(x, clf, target_class=1)
        )

    def test_default_target_class_matches_predicted_class(self, trained):
        """target_class=None falls back to the classifier's own prediction."""
        clf, data = trained
        x = data["X"][0]
        pred = int(clf.predict(x[np.newaxis])[0])
        expl = TimeSHAP(nsamples=100, seed=0)
        expl.fit(data["X"][:200])
        phi_default = expl.attribute(x, clf, target_class=None)
        phi_explicit = expl.attribute(x, clf, target_class=pred)
        assert np.array_equal(phi_default, phi_explicit)

    def test_works_without_fit_using_zero_baseline(self, trained):
        """Without fit(), a zero average-event baseline is used (still valid)."""
        clf, data = trained
        x = data["X"][0]
        phi = TimeSHAP(nsamples=100, seed=0).attribute(x, clf, target_class=1)
        assert phi.shape == x.shape
        assert np.all(np.isfinite(phi))
