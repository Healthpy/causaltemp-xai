"""Tests for the Axis-C metrics (validity flip-rate, proximity, sparsity, OOD)."""

from __future__ import annotations

import numpy as np

from causaltemp_xai.metrics.axis_c import (
    ood_plausibility,
    proximity,
    sparsity,
    validity,
)


class _LabelModel:
    """Mock classifier returning a fixed label per batch row (by order)."""

    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def predict(self, X):
        X = np.asarray(X)
        return self.labels[: X.shape[0]]


class TestValidity:
    def test_flip_rate_matches_manual(self):
        cfs = np.zeros((4, 3, 2))  # 4 instances, shape irrelevant to the mock
        model = _LabelModel([1, 1, 0, 1])
        # 3 of 4 predicted as the target class 1 → 0.75
        assert validity(cfs, model, target_class=1) == 0.75

    def test_all_valid(self):
        cfs = np.zeros((5, 4, 3))
        model = _LabelModel([1, 1, 1, 1, 1])
        assert validity(cfs, model, target_class=1) == 1.0

    def test_none_valid(self):
        cfs = np.zeros((3, 4, 3))
        model = _LabelModel([0, 0, 0])
        assert validity(cfs, model, target_class=1) == 0.0

    def test_single_instance_promoted(self):
        cf = np.zeros((4, 3))  # single (T, k)
        model = _LabelModel([1])
        assert validity(cf, model, target_class=1) == 1.0

    def test_accepts_plain_callable(self):
        cfs = np.zeros((2, 4, 3))
        model = lambda X: np.array([1, 0])  # noqa: E731
        assert validity(cfs, model, target_class=1) == 0.5


class TestProximity:
    def test_l1_known_vector(self):
        x = np.zeros((2, 3))
        x_cf = x.copy()
        x_cf[0, 0] = 3.0
        assert proximity(x, x_cf, norm="l1") == 3.0

    def test_l2_known_vector(self):
        x = np.zeros((2, 3))
        x_cf = x.copy()
        x_cf[0, 0] = 3.0
        x_cf[1, 2] = 4.0
        assert proximity(x, x_cf, norm="l2") == 5.0  # sqrt(9 + 16)

    def test_identical_is_zero(self):
        x = np.random.default_rng(0).normal(size=(5, 4))
        assert proximity(x, x.copy(), norm="l1") == 0.0


class TestSparsity:
    def test_fraction_unchanged(self):
        x = np.zeros((2, 3))  # 6 features
        x_cf = x.copy()
        x_cf[0, 0] = 1.0  # one feature altered
        assert sparsity(x, x_cf) == 5 / 6

    def test_identical_is_one(self):
        x = np.random.default_rng(1).normal(size=(3, 3))
        assert sparsity(x, x.copy()) == 1.0


class TestOOD:
    def test_returns_finite_scalar_for_single(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(50, 6, 4))
        x_cf = rng.normal(size=(6, 4))
        score = ood_plausibility(X_train, x_cf)
        assert np.isscalar(score) and np.isfinite(score)

    def test_returns_finite_array_for_batch(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(50, 6, 4))
        cfs = rng.normal(size=(5, 6, 4))
        scores = ood_plausibility(X_train, cfs)
        assert scores.shape == (5,) and np.all(np.isfinite(scores))
