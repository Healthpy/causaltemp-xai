"""The taxonomy is enforced, not asserted (2026-08-03).

"No metric stands alone" is a claim about *every* metric the benchmark reports,
so it has to be checked against what the pipeline actually emits rather than
against a hand-maintained list that drifts. These tests read the real
``evaluate_method`` / ``compute_axis_*`` outputs and fail if any key in them has
no axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism
from causaltemp_xai.metrics.taxonomy import AXES, AXIS_METRICS, AXIS_OF, axis_of

#: Bookkeeping / provenance keys that are not metrics and so need no axis.
NON_METRIC_KEYS = frozenset(
    {
        "n",
        "benchmark",
        "classifier",
        "method",
        "seed",
        "git_commit",
        "git_dirty",
        "n_scorable",
        "frac_no_intervention",
        "schedule_mode",
        "horizon",
        "t0",
        "t_label",
        "label_horizon",
    }
)


class TestTaxonomyShape:
    def test_three_axes(self):
        assert set(AXES) == {"A", "B", "C"}

    def test_axis_a_is_per_dataset_and_the_others_per_method(self):
        """The relabelling did not make the axes parallel — A still scores the
        benchmark, B and C score methods. Losing this distinction is the
        category error general_plan.md §5 warns about."""
        assert AXES["A"][1] == "dataset"
        assert AXES["B"][1] == "method"
        assert AXES["C"][1] == "method"

    def test_no_metric_claimed_by_two_axes(self):
        seen: dict[str, str] = {}
        for axis, metrics in AXIS_METRICS.items():
            for m in metrics:
                assert m not in seen, f"{m} claimed by both {seen.get(m)} and {axis}"
                seen[m] = axis

    def test_axis_of_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError, match="belongs to no axis"):
            axis_of("not_a_metric")

    def test_deleted_axis_a_metrics_are_gone(self):
        """The concept metrics were removed with the axis; nothing may re-home
        them silently."""
        for dead in ("icc", "icc_latent", "mig", "dci", "mcc", "mcc_concept"):
            assert dead not in AXIS_OF


class TestEveryEmittedMetricHasAnAxis:
    """The load-bearing check: walk the pipeline's real output keys."""

    @staticmethod
    def _fixture(T=25, k=4, n=6, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.uniform(-0.3, 0.3, (k, k))
        X = np.zeros((n, T, k))
        for i in range(n):
            for t in range(1, T):
                X[i, t] = A @ X[i, t - 1] + rng.laplace(0, 0.05, k)
        return X, LinearMechanism([A]), np.zeros((k, k, 1))

    def test_evaluate_method_keys_all_have_an_axis(self):
        from causaltemp_xai.eval import evaluate_method

        X, mech, graph = self._fixture()
        CFs = X + 0.4

        class _Clf:
            def predict(self, Z):
                return np.ones(len(np.atleast_3d(Z)), dtype=int)

        out = evaluate_method(_Clf(), CFs, X, X, graph, mech, target_class=1)
        orphans = [k for k in out if k not in AXIS_OF and k not in NON_METRIC_KEYS]
        assert not orphans, f"metrics emitted with no axis: {orphans}"

    def test_compute_axis_a_keys_are_axis_a(self):
        from causaltemp_xai.metrics import compute_axis_a

        _, _, graph = self._fixture()
        out = compute_axis_a(graph, graph)
        for key in out:
            if key in NON_METRIC_KEYS:
                continue
            assert AXIS_OF.get(key.lower(), "A") == "A", f"{key} is not Axis A"

    def test_pns_direction_keys_all_have_an_axis(self):
        from causaltemp_xai.benchmarks.structural_cf import structural_counterfactual
        from causaltemp_xai.metrics.pns import pns_direction

        X, mech, _ = self._fixture()
        CFs = np.stack([structural_counterfactual(x, mech, 5, 0, 3.0) for x in X])

        class _Clf:
            def predict(self, Z):
                return np.ones(len(np.atleast_3d(Z)), dtype=int)

        out = pns_direction(X, CFs, _Clf(), mech, theta=0.0, target_class=1)
        orphans = [k for k in out if k not in AXIS_OF and k not in NON_METRIC_KEYS]
        assert not orphans, f"PNS keys with no axis: {orphans}"

    def test_every_declared_axis_c_metric_is_actually_emitted(self):
        """The reverse direction, and the one that was missing.

        The forward check (emitted -> has an axis) passed all day while
        ``scm_noise_plausibility`` sat declared-but-unreachable: computed only
        inside an uncalled helper, so it never reached a result file. A
        taxonomy that lists metrics the pipeline does not produce is as wrong
        as one that misses metrics it does.
        """
        from causaltemp_xai.eval import evaluate_method
        from causaltemp_xai.metrics.pns import pns_direction

        X, mech, graph = self._fixture()
        CFs = X + 0.4

        class _Clf:
            def predict(self, Z):
                return np.ones(len(np.atleast_3d(Z)), dtype=int)

        emitted = set(evaluate_method(_Clf(), X, CFs, X, graph, mech, target_class=1)) | set(
            pns_direction(X, CFs, _Clf(), mech, theta=0.0, target_class=1)
        )
        # These two are produced by the experiments layer (per_instance_records
        # / aggregate_method_row), not by evaluate_method, so name them rather
        # than silently exempting anything absent.
        from_experiments_layer = {"trsi", "scm_noise_plausibility"}
        # M4g (2026-08-06): discovered-graph CF-faith on Tier 2
        # is produced by experiments/07b_discovered_graph_real.py's
        # score_method_discovered, a real (numbered-module) phase, not by
        # evaluate_method/pns_direction either. Checked against that real
        # producer in test_the_07b_discovered_graph_layer_emits_what_it_is_credited_with
        # below, same discipline as the two exemptions above.
        from_07b_discovered_graph = {
            "cf_faith_discovered_rollout_mean",
            "cf_faith_discovered_rollout_ci_lo",
            "cf_faith_discovered_rollout_ci_hi",
            "cf_faith_discovered_pearl_mean",
            "cf_faith_discovered_pearl_ci_lo",
            "cf_faith_discovered_pearl_ci_hi",
        }

        missing = (
            set(AXIS_METRICS["C"]) - emitted - from_experiments_layer - from_07b_discovered_graph
        )
        assert not missing, f"declared on Axis C but never emitted: {sorted(missing)}"

    def test_the_experiments_layer_emits_what_it_is_credited_with(self):
        """...and those two are checked against the real producer, so the
        exemption above cannot become a hiding place."""
        from experiments._common import aggregate_method_row, per_instance_records

        X, mech, graph = self._fixture()
        CFs = X + 0.4
        rows = per_instance_records(
            "t",
            "lstm",
            "m",
            X,
            CFs,
            graph,
            mech,
            preds=np.ones(len(X), dtype=int),
            noise_scale=0.1,
        )
        agg = aggregate_method_row("t", "lstm", "m", rows)
        for key in ("trsi", "scm_noise_plausibility"):
            assert key in agg, f"{key} missing from aggregate_method_row"
            assert agg[key] is not None, f"{key} present but None"

    def test_the_07b_discovered_graph_layer_emits_what_it_is_credited_with(self):
        """M4g: ...and the discovered-graph exemption is checked against its
        real producer too, same discipline as
        test_the_experiments_layer_emits_what_it_is_credited_with above."""
        import importlib

        from causaltemp_xai.benchmarks.generator import NlinearSCMT
        from causaltemp_xai.methods.causal import DYNOTEARS

        _phase07b = importlib.import_module("experiments.07b_discovered_graph_real")

        gen = NlinearSCMT(k=4, L=1, sparsity=0.3, T=20, N=60, seed=0)
        X = gen.generate()["X"]
        mechanisms = [
            DYNOTEARS(k=4, p=1).fit(X).to_linear_mechanism(),
            DYNOTEARS(k=4, p=1).fit(X).to_linear_mechanism(),
        ]
        row = _phase07b.score_method_discovered(X[:5], X[:5] + 0.4, mechanisms, seed=0)
        for key in (
            "cf_faith_discovered_rollout_mean",
            "cf_faith_discovered_rollout_ci_lo",
            "cf_faith_discovered_rollout_ci_hi",
            "cf_faith_discovered_pearl_mean",
            "cf_faith_discovered_pearl_ci_lo",
            "cf_faith_discovered_pearl_ci_hi",
        ):
            assert key in row, f"{key} missing from score_method_discovered"
            assert row[key] is not None, f"{key} present but None"

    def test_the_audit_metrics_live_on_axis_c(self):
        """CF-faith, the model-vs-world terms and do-complexity were previously
        described as a gate / an audit / a diagnostic — none of them an axis.
        They are Axis C now."""
        for m in ("cf_faith_rollout_hard", "delta_total", "do_complexity_mean", "frac_vacuous"):
            assert axis_of(m) == "C"
