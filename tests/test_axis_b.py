"""Adversarial tests for Axis-B structural-diagnostic metrics (M1, objective O1).

Axis B is computed once per *dataset* in Phase 01 (no per-method score), but each
graph metric must still discriminate a correct estimate from a wrong one: a metric
that scores a scrambled graph the same as the ground truth measures nothing.

Contract per metric: (a) penalise a constructed *wrong* graph estimate and
(b) reward the ground-truth (or a compliant) estimate. Governed by R6.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.metrics.axis_b import (
    graph_auc,
    graph_error_decomposition,
    lag_accuracy,
    shd,
    tv_confounding,
)


def _true_lagged_graph() -> np.ndarray:
    """A small (k=3, max_lag=2) binary lagged adjacency with three edges."""
    g = np.zeros((3, 3, 2), dtype=int)
    g[0, 1, 0] = 1  # 0 -> 1 at lag 1
    g[1, 2, 1] = 1  # 1 -> 2 at lag 2
    g[2, 0, 0] = 1  # 2 -> 0 at lag 1
    return g


# ---------------------------------------------------------------------------
# SHD — Structural Hamming Distance
# ---------------------------------------------------------------------------


class TestSHDAdversarial:
    def test_flags_fully_wrong_graph(self):
        g = _true_lagged_graph()
        pred = 1 - g  # every edge decision flipped
        assert shd(g, pred) == g.size

    def test_passes_identical_graph(self):
        g = _true_lagged_graph()
        assert shd(g, g) == 0

    def test_counts_each_wrong_edge_once(self):
        g = _true_lagged_graph()
        pred = g.copy()
        pred[0, 2, 0] = 1  # one spurious edge (false positive)
        pred[0, 1, 0] = 0  # one dropped edge (false negative)
        assert shd(g, pred) == 2


# ---------------------------------------------------------------------------
# Lag accuracy — right edge at the right lag
# ---------------------------------------------------------------------------


class TestLagAccuracyAdversarial:
    def test_flags_correct_edges_wrong_lag(self):
        g = _true_lagged_graph()
        # Same (i, j) edges, but each placed at the *other* lag.
        wrong = np.zeros_like(g)
        wrong[0, 1, 1] = 1
        wrong[1, 2, 0] = 1
        wrong[2, 0, 1] = 1
        assert lag_accuracy(g, wrong) == 0.0

    def test_passes_correct_lags(self):
        g = _true_lagged_graph()
        assert lag_accuracy(g, g) == 1.0

    def test_partial_credit(self):
        g = _true_lagged_graph()
        pred = g.copy()
        pred[0, 1, 0] = 0
        pred[0, 1, 1] = 1  # move one of three edges to the wrong lag
        assert lag_accuracy(g, pred) == pytest.approx(2 / 3)

    def test_no_true_edges_is_nan(self):
        empty = np.zeros((3, 3, 2), dtype=int)
        assert np.isnan(lag_accuracy(empty, empty))


# ---------------------------------------------------------------------------
# Graph AUC — edge-detection ranking
# ---------------------------------------------------------------------------


class TestGraphAUCAdversarial:
    def _truth(self) -> np.ndarray:
        return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def test_passes_aligned_scores(self):
        truth = self._truth()
        scores = truth.astype(float)  # edges rank strictly above non-edges
        assert graph_auc(truth, scores) == 1.0

    def test_flags_inverted_scores(self):
        truth = self._truth()
        scores = (1 - truth).astype(float)  # anti-correlated ranking
        assert graph_auc(truth, scores) == 0.0

    def test_single_class_is_nan(self):
        truth = np.zeros((3, 3), dtype=int)  # no positive edges
        assert np.isnan(graph_auc(truth, np.random.default_rng(0).normal(size=(3, 3))))


# ---------------------------------------------------------------------------
# TV-Confounding — marginal divergence of non-adjacent channel pairs
# ---------------------------------------------------------------------------


class TestTVConfoundingAdversarial:
    def test_low_for_matching_marginals(self):
        # Two non-adjacent channels drawn from the same distribution: a
        # confounding score should be near zero.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 10, 2))
        adj = np.zeros((2, 2))
        assert tv_confounding(X, adj) < 0.2

    def test_high_for_divergent_marginals(self):
        # Same non-adjacent pair, but the marginals are pushed far apart: the
        # unexplained (non-edge) association must register as large TV.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 10, 2))
        X[:, :, 1] += 100.0
        adj = np.zeros((2, 2))
        assert tv_confounding(X, adj) > 0.8

    def test_divergent_exceeds_matching(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(400, 10, 2))
        shifted = base.copy()
        shifted[:, :, 1] += 100.0
        adj = np.zeros((2, 2))
        assert tv_confounding(shifted, adj) > tv_confounding(base, adj)

    def test_fully_connected_has_no_pairs(self):
        X = np.zeros((5, 10, 2))
        adj = np.ones((2, 2))  # no non-adjacent pair exists
        assert tv_confounding(X, adj) == 0.0


# ---------------------------------------------------------------------------
# Graph-error decomposition — arithmetic identity
# ---------------------------------------------------------------------------


class TestGraphErrorDecomposition:
    def test_splits_graph_vs_propagation_error(self):
        d = graph_error_decomposition(0.9, 0.6)
        assert d["cf_faith_gt"] == 0.9
        assert d["cf_faith_inferred"] == 0.6
        assert d["graph_error"] == pytest.approx(0.3)
        assert d["propagation_error"] == pytest.approx(0.1)

    def test_no_graph_error_when_estimates_agree(self):
        d = graph_error_decomposition(0.8, 0.8)
        assert d["graph_error"] == 0.0
