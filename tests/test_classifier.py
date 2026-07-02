"""Tests for the LSTMClassifier wrapper.

Verifies:
- Overfits a tiny synthetic set to high train accuracy in few epochs.
- predict / predict_proba shapes; probabilities sum to 1.
- save/load round-trip yields identical predictions.
- torch_logits returns a grad-enabled tensor and backprop reaches the input.
- Shape contract: (T,k) / (N,T,k) flow through the classifier and CFfaith
  with no manual transpose at the call site.
"""

from __future__ import annotations

import numpy as np
import torch

from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.metrics.cf_faith import CFfaith


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _separable_dataset(n=64, T=12, k=3, seed=0):
    """Two well-separated classes: class 1 has a constant positive offset."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.normal(0.0, 0.3, (half, T, k))
    X1 = rng.normal(2.0, 0.3, (n - half, T, k))
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(half), np.ones(n - half)]).astype(int)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _small_clf(k=3, **kw):
    defaults = dict(
        n_inputs=k,
        hidden_size=16,
        num_layers=2,
        dropout=0.0,
        lr=5e-3,
        batch_size=32,
        max_epochs=80,
        patience=80,
        seed=0,
    )
    defaults.update(kw)
    return LSTMClassifier(**defaults)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestFit:
    def test_overfits_tiny_set(self):
        X, y = _separable_dataset(n=64, T=12, k=3, seed=1)
        clf = _small_clf(k=3, target_acc=0.9).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_fit_with_validation_early_stops(self):
        X, y = _separable_dataset(n=80, T=10, k=3, seed=2)
        clf = _small_clf(k=3, patience=5).fit(X[:60], y[:60], X[60:], y[60:])
        # Still classifies the separable val set well after restoring best weights.
        assert clf.score(X[60:], y[60:]) >= 0.9


# ---------------------------------------------------------------------------
# Inference shapes
# ---------------------------------------------------------------------------


class TestInferenceShapes:
    def test_predict_proba_shape_and_sums_to_one(self):
        X, y = _separable_dataset(n=32, T=10, k=3, seed=3)
        clf = _small_clf(k=3, max_epochs=5).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (32, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_batch_shape(self):
        X, y = _separable_dataset(n=20, T=10, k=3, seed=4)
        clf = _small_clf(k=3, max_epochs=5).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_single_instance_returns_scalar(self):
        X, y = _separable_dataset(n=20, T=10, k=3, seed=5)
        clf = _small_clf(k=3, max_epochs=5).fit(X, y)
        label = clf.predict(X[0])  # (T, k)
        assert isinstance(label, int)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_identical_predictions(self, tmp_path):
        X, y = _separable_dataset(n=40, T=10, k=3, seed=6)
        clf = _small_clf(k=3, max_epochs=10).fit(X, y)
        path = tmp_path / "lstm.pt"
        clf.save(path)
        assert path.exists()

        reloaded = LSTMClassifier.load(path)
        np.testing.assert_array_equal(clf.predict(X), reloaded.predict(X))
        np.testing.assert_allclose(
            clf.predict_proba(X), reloaded.predict_proba(X), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_torch_logits_backprop_reaches_input(self):
        X, y = _separable_dataset(n=20, T=10, k=3, seed=7)
        clf = _small_clf(k=3, max_epochs=5).fit(X, y)

        x = torch.tensor(X[0], dtype=torch.float32, requires_grad=True)  # (T,k)
        logits = clf.torch_logits(x)
        assert logits.shape == (1, 2)
        assert logits.requires_grad
        logits[0, 1].backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)


# ---------------------------------------------------------------------------
# Shape contract: (T,k) everywhere, no manual transpose
# ---------------------------------------------------------------------------


class TestShapeContract:
    def test_tk_flows_through_classifier_and_cffaith(self):
        """One (T,k) instance flows through predict/proba/torch_logits and
        CFfaith.score with no transpose at the call site."""
        from causaltemp_xai.benchmark.generator import LinearSCMT

        gen = LinearSCMT(k=3, L=1, T=15, N=40, seed=8)
        data = gen.generate()
        X, graph, mechanism = data["X"], data["graph"], data["mechanism"]

        clf = _small_clf(k=3, max_epochs=5).fit(X, data["Y"])

        x_tk = X[0]  # (T, k) — public layout, no transpose
        assert isinstance(clf.predict(x_tk), int)
        assert clf.predict_proba(x_tk).shape == (1, 2)

        xt = torch.tensor(x_tk, dtype=torch.float32)
        assert clf.torch_logits(xt).shape == (1, 2)

        # CFfaith accepts the same (T, k) layout directly.
        scorer = CFfaith(tol=1e-3)
        result = scorer.score(x_tk, x_tk.copy(), 5, graph, mechanism)
        assert set(result.keys()) == {"hard", "soft"}
