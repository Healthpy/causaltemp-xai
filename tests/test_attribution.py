"""Tests for the Integrated-Gradients foil and deletion/insertion curves."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causaltemp_xai.attribution import (
    deletion_curve,
    insertion_curve,
    integrated_gradients,
)
from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.classifiers import TCNClassifier


@pytest.fixture(scope="module")
def trained():
    gen = LinearSCMT(k=5, L=1, T=20, N=240, seed=0)
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    clf = TCNClassifier(
        n_inputs=5,
        n_levels=2,
        n_channels=16,
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
