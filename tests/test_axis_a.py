"""Adversarial tests for Axis-A structural-diagnostic metrics (M1, objective O1).

Axis A is computed once per *dataset* in Phase 01 (no per-method score), but each
graph metric must still discriminate a correct estimate from a wrong one: a metric
that scores a scrambled graph the same as the ground truth measures nothing.

Contract per metric: (a) penalise a constructed *wrong* graph estimate and
(b) reward the ground-truth (or a compliant) estimate. Governed by R6.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
from causaltemp_xai.metrics.axis_a import (
    graph_auc,
    graph_error_decomposition,
    lag_accuracy,
    lagged_edge_f1,
    residual_dependence,
    shd,
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

    @pytest.mark.parametrize("constant", [0.0, 1.0, -3.5])
    def test_constant_scores_are_nan(self, constant):
        """A score matrix with no variation expresses no ranking at all.

        Returning 0.5 launders "this method produced nothing" into "this method
        performed at chance", which is a substantive empirical claim the data
        does not support. NaN is the honest answer: the metric is undefined,
        not satisfied-at-chance.
        """
        truth = self._truth()  # both classes present, so y_true is not the issue
        scores = np.full(truth.shape, constant, dtype=float)
        assert np.isnan(graph_auc(truth, scores)), (
            f"constant score matrix (all {constant}) must be NaN, "
            f"got {graph_auc(truth, scores)}"
        )


# ---------------------------------------------------------------------------
# Residual dependence — unexplained association between non-adjacent channels
# (replaces tv_confounding, metric-quality fix #1 2026-07-18)
# ---------------------------------------------------------------------------


class TestResidualDependenceAdversarial:
    def test_low_for_independent_channels(self):
        # Two independent non-adjacent channels: no unexplained association.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 10, 2))
        adj = np.zeros((2, 2))
        assert residual_dependence(X, adj) < 0.1

    def test_marginal_shift_is_not_confounding(self):
        # The retired TV formulation's false positive: a mean shift changes
        # the marginals but creates no dependence — must stay low.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 10, 2))
        X[:, :, 1] += 100.0
        adj = np.zeros((2, 2))
        assert residual_dependence(X, adj) < 0.1

    def test_flags_shared_latent_confounder(self):
        # A genuine confounder: a shared latent added to both non-adjacent
        # channels induces dependence and must score high.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 10, 2))
        latent = rng.normal(size=(200, 10))
        X[:, :, 0] += 2.0 * latent
        X[:, :, 1] += 2.0 * latent
        adj = np.zeros((2, 2))
        assert residual_dependence(X, adj) > 0.5

    def test_mechanism_mode_confounder_free_scm_is_low(self):
        # On a confounder-free VAR the abducted residuals of non-adjacent
        # channels are independent noise — the diagnostic must read ~0
        # (the retired TV score read ~0.6 here, the motivating defect).
        rng = np.random.default_rng(2)
        A = np.array([[0.5, 0.0], [0.0, 0.5]])  # two disconnected channels
        mech = LinearMechanism([A])
        N, T, k = 60, 30, 2
        X = np.zeros((N, T, k))
        noise = rng.laplace(0, 0.1, (N, T, k))
        for t in range(1, T):
            X[:, t] = X[:, t - 1] @ A.T + noise[:, t]
        adj = np.zeros((2, 2))
        assert residual_dependence(X, adj, mechanism=mech) < 0.1

    def test_fully_connected_has_no_pairs(self):
        X = np.zeros((5, 10, 2))
        adj = np.ones((2, 2))  # no non-adjacent pair exists
        assert residual_dependence(X, adj) == 0.0


# ---------------------------------------------------------------------------
# Lagged-edge F1 — precision-aware companion to recall-only lag_accuracy
# (metric-quality fix #7 2026-07-18)
# ---------------------------------------------------------------------------


class TestLaggedEdgeF1Adversarial:
    def test_complete_graph_does_not_score_one(self):
        # The exploit lag_accuracy cannot catch: predicting every edge at
        # every lag gives LagAcc = 1.0 but must not give F1 = 1.0.
        g = _true_lagged_graph()
        complete = np.ones_like(g)
        assert lag_accuracy(g, complete) == 1.0  # the documented gap
        assert lagged_edge_f1(g, complete) < 1.0

    def test_perfect_graph_scores_one(self):
        g = _true_lagged_graph()
        assert lagged_edge_f1(g, g) == 1.0

    def test_empty_prediction_scores_zero(self):
        g = _true_lagged_graph()
        assert lagged_edge_f1(g, np.zeros_like(g)) == 0.0

    def test_no_true_edges_is_nan(self):
        empty = np.zeros((3, 3, 2), dtype=int)
        assert np.isnan(lagged_edge_f1(empty, empty))


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
