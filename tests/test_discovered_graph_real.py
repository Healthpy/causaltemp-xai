"""R6 adversarial tests for M4g's discovered-graph CF-faith on Tier 2
(2026-08-06).

Genuine Tier 2 data has no ground truth, so nothing there can validate
whether the discovered-graph pipeline behaves sanely. This file uses
**synthetic data as a stand-in "fake real data"**: a `NlinearSCMT` instance
whose true graph/mechanism are known, so a constructed CF's *true*
faithfulness can be compared against what the discovered-graph pipeline
(DYNOTEARS ensemble -> `to_linear_mechanism` -> `score_method_discovered`)
reports for the *same* CF, scored only against the *inferred* mechanisms.

Per R6: construct a CF that should pass and one that should
fail, and verify the metric actually distinguishes them, before any table
cites a `cf_faith_discovered_*` number.
"""

from __future__ import annotations

import importlib

import numpy as np

from causaltemp_xai.benchmarks.generator import NlinearSCMT
from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual

_phase07b = importlib.import_module("experiments.07b_discovered_graph_real")
score_method_discovered = _phase07b.score_method_discovered
_fit_mechanism_ensemble = _phase07b._fit_mechanism_ensemble


def _make_scm(k=4, L=1, T=25, N=150, seed=0):
    gen = NlinearSCMT(k=k, L=L, sparsity=0.3, T=T, N=N, seed=seed, hidden=8)
    data = gen.generate(burn_in=20)
    return data["X"], data["graph"], data["mechanism"]


def _mechanism_ensemble(X, k, L, B=3, seed=0):
    return _fit_mechanism_ensemble(X, k, L, B, seed=seed)


class TestDiscoveredGraphCFFaithAdversarial:
    def test_truly_faithful_cf_scores_reasonably_high(self):
        """An oracle CF (cf_faith = 1.0 against the TRUE mechanism, by
        construction) should score *reasonably* high against a
        well-recovered ensemble of INFERRED mechanisms -- not necessarily
        1.0 (the mechanisms are only approximate), but clearly not near-zero
        either. Sanity that the construction is not systematically broken."""
        X, graph, mech = _make_scm(N=200, seed=0)
        X_sel = X[:15]
        t0, node = 5, 0
        cfs = np.stack(
            [
                structural_counterfactual(
                    x, mech, t0, node, float(x[t0, node]) + 1.0, noiseless=True
                )
                for x in X_sel
            ]
        )

        k, L = graph.shape[0], graph.shape[2]
        mechanisms = _mechanism_ensemble(X, k, L, B=3, seed=0)
        row = score_method_discovered(X_sel, cfs, mechanisms, seed=0)

        assert row["cf_faith_discovered_rollout_mean"] > 0.5
        assert row["ensemble_b"] == 3

    def test_adversarial_cf_scores_low_even_under_approximate_mechanism(self):
        """A CF that is deliberately mechanism-inconsistent (random noise
        after t0, unrelated to any structural equation) must score low even
        though the mechanism it's checked against is only an approximation --
        confirms the metric can still reject a bad CF."""
        X, graph, _mech = _make_scm(N=200, seed=0)
        X_sel = X[:15]
        t0 = 5
        rng = np.random.default_rng(0)
        cfs = X_sel.copy()
        # Large, structurally arbitrary post-t0 perturbation -- not derived
        # from the mechanism in any way.
        cfs[:, t0:, :] += rng.normal(scale=5.0, size=cfs[:, t0:, :].shape)

        k, L = graph.shape[0], graph.shape[2]
        mechanisms = _mechanism_ensemble(X, k, L, B=3, seed=0)
        row = score_method_discovered(X_sel, cfs, mechanisms, seed=0)

        assert row["cf_faith_discovered_rollout_mean"] < 0.3

    def test_faithful_scores_higher_than_adversarial(self):
        """The sharpest single check: same data, same mechanisms, only the
        CF differs -- the metric must separate the two directionally, not
        just clear independent absolute thresholds."""
        X, graph, mech = _make_scm(N=200, seed=1)
        X_sel = X[:15]
        t0, node = 5, 0
        k, L = graph.shape[0], graph.shape[2]
        mechanisms = _mechanism_ensemble(X, k, L, B=3, seed=1)

        faithful_cfs = np.stack(
            [
                structural_counterfactual(
                    x, mech, t0, node, float(x[t0, node]) + 1.0, noiseless=True
                )
                for x in X_sel
            ]
        )
        rng = np.random.default_rng(1)
        adversarial_cfs = X_sel.copy()
        adversarial_cfs[:, t0:, :] += rng.normal(scale=5.0, size=adversarial_cfs[:, t0:, :].shape)

        faithful_row = score_method_discovered(X_sel, faithful_cfs, mechanisms, seed=1)
        adversarial_row = score_method_discovered(X_sel, adversarial_cfs, mechanisms, seed=1)

        assert (
            faithful_row["cf_faith_discovered_rollout_mean"]
            > adversarial_row["cf_faith_discovered_rollout_mean"]
        )

    def test_zero_intervention_cf_scores_near_one_regardless_of_mechanism(self):
        """A CF identical to its factual (no edit at all) is trivially
        faithful under ANY mechanism -- true by construction, a cheap
        regression pin independent of how well DYNOTEARS recovered anything."""
        X, graph, _mech = _make_scm(N=100, seed=0)
        X_sel = X[:10]
        cfs = X_sel.copy()  # zero intervention

        k, L = graph.shape[0], graph.shape[2]
        mechanisms = _mechanism_ensemble(X, k, L, B=3, seed=0)
        row = score_method_discovered(X_sel, cfs, mechanisms, seed=0)

        # derive_intervention_t lands at T-1 for a genuinely zero-edit CF
        # (nothing ever differs), which is CFfaith's own degeneracy gate --
        # both semantics return NaN, and nanmean over all-NaN is NaN. This
        # is the honest, already-established behavior elsewhere in this
        # codebase (see tests/test_auxiliary_methods.py::TestMeanSoftCfFaith
        # ::test_all_gated_returns_nan_not_zero) -- not special-cased here,
        # just asserted explicitly so a future reader isn't surprised by NaN.
        assert np.isnan(row["cf_faith_discovered_rollout_mean"])
        assert np.isnan(row["cf_faith_discovered_pearl_mean"])

    def test_empty_ensemble_is_rejected_by_bootstrap_ci_not_silently_wrong(self):
        """Degenerate input hygiene: an empty mechanism list must not
        silently produce a fabricated score."""
        X, _graph, _mech = _make_scm(N=50, seed=0)
        X_sel = X[:5]
        cfs = X_sel.copy()
        row = score_method_discovered(X_sel, cfs, mechanisms=[], seed=0)
        assert row["ensemble_b"] == 0
        assert np.isnan(row["cf_faith_discovered_rollout_mean"])
