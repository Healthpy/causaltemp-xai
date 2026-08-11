"""Tests for PCMCIplus (`tigramite`, installed as a PyPI dependency, M4h
2026-08-06, `DECISIONS.md`) -- the second self-graphing method, wired
specifically to answer `docs/risk_register.md` RISK-22's named gap.

`TestPCMCIPlus` mirrors `tests/test_dynotears.py::TestDYNOTEARS`'s shape for
the structurally different second method. `TestCrossMethodAgreement`
exercises the DYNOTEARS-vs-PCMCIplus density-match/SHD/AUC logic
`experiments/07_auxiliary_methods.py::run_cross_method_agreement` uses, on a
synthetic fixture with known ground truth (mirrors
`tests/test_auxiliary_methods.py::TestGraphQualitySweepEnsemble`'s own
`_make_scm` fixture style, rather than going through the file-system-backed
`load_dataset`/`get_config` path `run_cross_method_agreement` itself uses).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from causaltemp_xai.benchmarks.generator import NlinearSCMT
from causaltemp_xai.methods.causal import DYNOTEARS, PCMCIPlus

_phase07 = importlib.import_module("experiments.07_auxiliary_methods")


def _auc(y, s):
    """Rank AUC robust to ties (mean rank of positives). Mirrors
    `tests/test_dynotears.py`'s own helper (kept duplicated, not imported,
    to avoid cross-test-file coupling for a five-line helper)."""
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class TestPCMCIPlus:
    def test_inferred_graph_shapes(self):
        g = NlinearSCMT(k=4, L=1, sparsity=0.3, T=20, N=100, seed=0)
        m = PCMCIPlus(k=4, tau_max=1).fit(g.generate()["X"])
        adj, scores = m.inferred_graph(max_lag=1)
        assert adj.shape == (4, 4, 1)
        assert scores.shape == (4, 4, 1)
        assert adj.dtype == int
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        assert np.all(np.diagonal(scores[:, :, 0]) == 0.0)  # no self-loops

    def test_recovers_lag1_structure_above_chance(self):
        """Averaged over seeds, PCMCIplus ranks true edges above non-edges.
        Not asserting parity with DYNOTEARS's own ~0.9 -- PCMCIplus is
        constraint-based, a structurally different method, not expected to
        match a continuous-optimisation method's recovery quality on this
        near-linear SCM. See the real smoke_nl reading recorded in
        `DECISIONS.md` (M4h) for the actual measured number."""
        aucs = []
        for seed in range(3):
            g = NlinearSCMT(k=5, L=1, sparsity=0.3, T=25, N=400, seed=seed)
            m = PCMCIPlus(k=5, tau_max=1).fit(g.generate()["X"])
            _, scores = m.inferred_graph(max_lag=1)
            tg = g.graph[:, :, 0].astype(bool)
            off = ~np.eye(5, dtype=bool)
            aucs.append(_auc(tg[off].astype(int), scores[:, :, 0][off]))
        assert np.nanmean(aucs) > 0.6  # above chance

    def test_no_contemporaneous_edges_included(self):
        """`tau_min` is fixed at 1 internally (not exposed as a constructor
        parameter): confirms the adapter is actually excluding lag-0 links,
        not silently including them by accident."""
        g = NlinearSCMT(k=4, L=1, sparsity=0.3, T=20, N=100, seed=0)
        m = PCMCIPlus(k=4, tau_max=1).fit(g.generate()["X"])
        assert m.graph_.shape[2] == 2  # tau=0 and tau=1 slices exist
        assert np.all(m.graph_[:, :, 0] == "")  # tau=0 untested -- all empty

    def test_wrong_channel_count_raises(self):
        g = NlinearSCMT(k=4, L=1, T=20, N=20, seed=0)
        with pytest.raises(ValueError):
            PCMCIPlus(k=5, tau_max=1).fit(g.generate()["X"])

    def test_inferred_graph_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PCMCIPlus(k=4, tau_max=1).inferred_graph()

    def test_has_no_to_linear_mechanism(self):
        """Hard limitation, tested explicitly rather than left implicit:
        PCMCIplus cannot participate in M4g's Tier-2 discovered-mechanism
        pipeline (`ParCorr`'s test statistics are not rollout-usable
        regression coefficients). See `docs/risk_register.md` RISK-22."""
        assert not hasattr(PCMCIPlus, "to_linear_mechanism")


class TestCrossMethodAgreement:
    """DYNOTEARS-vs-PCMCIplus cross-method agreement check (M4h, RISK-22)."""

    def _make_scm(self, k=5, L=1, T=25, N=300, seed=0):
        g = NlinearSCMT(k=k, L=L, sparsity=0.3, T=T, N=N, seed=seed, hidden=8)
        data = g.generate(burn_in=20)
        return data["X"], data["graph"]

    def test_self_comparison_is_zero_shd(self):
        """Degenerate-anchor regression pin: comparing a method's graph
        against itself must be exactly SHD=0."""
        from causaltemp_xai.metrics.axis_a import shd

        X, graph = self._make_scm(seed=0)
        k, _, L = graph.shape
        n_true = int((graph > 0).sum())
        m = DYNOTEARS(k=k, p=L).fit(X)
        _, scores = m.inferred_graph(max_lag=L)
        adj = _phase07._density_matched_adjacency(scores, n_true)
        assert shd(adj, adj) == 0

    def test_genuinely_different_graphs_give_nonzero_shd(self):
        """A true graph vs. its fully-corrupted (density-matched random)
        counterpart must show nonzero SHD -- confirms the metric direction
        is sane before trusting it on real method output."""
        from causaltemp_xai.metrics.axis_a import shd

        _X, graph = self._make_scm(seed=0)
        true_bin = (graph > 0).astype(int)
        rng = np.random.default_rng(0)
        corrupted = _phase07._corrupt_graph(true_bin, frac=1.0, rng=rng)
        assert shd(true_bin, corrupted) > 0

    def test_dynotears_and_pcmciplus_both_report_auc_and_agreement(self):
        """Real end-to-end run: both methods fit on the same data, each
        reports its own AUC against the true graph, plus a non-negative
        SHD-between-methods -- no numeric agreement threshold pre-committed
        (that is an empirical smoke-scale finding to record, per M4h's own
        validation step, not something to assume in a unit test)."""
        from causaltemp_xai.metrics.axis_a import graph_auc, shd

        X, graph = self._make_scm(seed=0)
        k, _, L = graph.shape
        true_bin = (graph > 0).astype(int)
        n_true = int(true_bin.sum())

        dyn = DYNOTEARS(k=k, p=L).fit(X)
        _, dyn_scores = dyn.inferred_graph(max_lag=L)
        dyn_adj = _phase07._density_matched_adjacency(dyn_scores, n_true)

        pcmci = PCMCIPlus(k=k, tau_max=L).fit(X)
        _, pcmci_scores = pcmci.inferred_graph(max_lag=L)
        pcmci_adj = _phase07._density_matched_adjacency(pcmci_scores, n_true)

        shd_between = shd(dyn_adj, pcmci_adj)
        dyn_auc = graph_auc(true_bin, dyn_scores)
        pcmci_auc = graph_auc(true_bin, pcmci_scores)

        assert shd_between >= 0
        assert 0.0 <= dyn_auc <= 1.0
        assert 0.0 <= pcmci_auc <= 1.0
