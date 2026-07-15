"""Tests for DYNOTEARS (genuine vendored McKinsey CausalNex solver).

Unlike the CITRIS tests, these DO assert graph-recovery quality: DYNOTEARS is a
classical, deterministic structure learner that reliably recovers this
benchmark's near-linear lag-1 structure at smoke scale (mean edge-ranking AUC
~0.9), so a quality check is a legitimate guard, not a flaky one.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.generator import NlinearSCMT
from causaltemp_xai.methods.causal import DYNOTEARS


def _auc(y, s):
    """Rank AUC robust to ties (mean rank of positives)."""
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class TestDYNOTEARS:
    def test_uses_upstream_solver(self):
        """The solver must be the genuine vendored CausalNex function."""
        from causaltemp_xai.methods.causal.dynotears import _load_upstream_dynotears
        mod = _load_upstream_dynotears()
        assert hasattr(mod, "_learn_dynamic_structure")

    def test_inferred_graph_shapes(self):
        g = NlinearSCMT(k=4, L=1, sparsity=0.3, T=20, N=100, seed=0)
        m = DYNOTEARS(k=4, p=1).fit(g.generate()["X"])
        adj, scores = m.inferred_graph(max_lag=1)
        assert adj.shape == (4, 4, 1)
        assert scores.shape == (4, 4, 1)
        assert adj.dtype == int
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        assert np.all(np.diagonal(scores[:, :, 0]) == 0.0)  # no self-loops

    def test_recovers_lag1_structure_above_chance(self):
        """Averaged over seeds, DYNOTEARS ranks true edges above non-edges."""
        aucs = []
        for seed in range(3):
            g = NlinearSCMT(k=5, L=1, sparsity=0.3, T=25, N=400, seed=seed)
            m = DYNOTEARS(k=5, p=1).fit(g.generate()["X"])
            _, scores = m.inferred_graph(max_lag=1)
            tg = g.graph[:, :, 0].astype(bool)
            off = ~np.eye(5, dtype=bool)
            aucs.append(_auc(tg[off].astype(int), scores[:, :, 0][off]))
        assert np.nanmean(aucs) > 0.7  # well above chance (typically ~0.9)

    def test_no_instantaneous_edges_learned(self):
        """The benchmark has no lag-0 edges, so intra-slice W stays near zero."""
        g = NlinearSCMT(k=4, L=1, sparsity=0.3, T=20, N=200, seed=0)
        m = DYNOTEARS(k=4, p=1).fit(g.generate()["X"])
        assert np.abs(m.w_est_).max() < 0.5  # much smaller than lagged effects

    def test_wrong_channel_count_raises(self):
        g = NlinearSCMT(k=4, L=1, T=20, N=20, seed=0)
        with pytest.raises(ValueError):
            DYNOTEARS(k=5, p=1).fit(g.generate()["X"])

    def test_inferred_graph_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            DYNOTEARS(k=4, p=1).inferred_graph()
