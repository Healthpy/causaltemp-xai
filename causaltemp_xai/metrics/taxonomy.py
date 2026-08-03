"""The metric → axis map. Every reported metric belongs to exactly one axis.

Three axes, fixed 2026-08-03 (``DECISIONS.md``):

======  ==========================  =================
axis    scope                       unit of analysis
======  ==========================  =================
``A``   structure (graph)           per **dataset**
``B``   robustness                  per method
``C``   counterfactual quality      per method
======  ==========================  =================

**The axes are not parallel columns.** Axis A scores the *benchmark* — it is
computed once per dataset in Phase 01 and characterises the data-generating
process, not any explainer. Axes B and C score *methods*. A table that prints A
beside B and C as though all three were per-method scores is a category error
(``general_plan.md`` §5), and the relabelling that produced this taxonomy did
not license it — only the naming changed.

**Why this module exists.** Before 2026-08-03, CF-faith was "a gate", the PNS
terms were "an audit", do-complexity was "a diagnostic" and the graph metrics
were "not an axis" — four different words for four things that all appear in
result tables, with no single place saying where each belongs. That made the
reporting surface impossible to state in one sentence and easy to drift.
:data:`AXIS_OF` is now the one place, and
``tests/test_taxonomy.py::test_no_metric_stands_alone`` fails if a public metric
is added without a home.
"""

from __future__ import annotations

__all__ = ["AXES", "AXIS_METRICS", "AXIS_OF", "axis_of"]

#: Axis letter → (short name, unit of analysis).
AXES: dict[str, tuple[str, str]] = {
    "A": ("structure", "dataset"),
    "B": ("robustness", "method"),
    "C": ("counterfactual quality", "method"),
}

#: Axis letter → the metric keys it owns, as they appear in result files.
#:
#: Keys are the *reported* names (what lands in ``summary.json`` /
#: ``per_instance.csv``), not the Python function names, because the taxonomy
#: has to be checkable against what a reader actually sees in a table.
AXIS_METRICS: dict[str, tuple[str, ...]] = {
    "A": (
        "shd",
        "lag_accuracy",
        "lagged_edge_f1",
        "graph_auc",
        "residual_dependence",
        "graph_error",
        "propagation_error",
    ),
    "B": (
        "shift_vr",
        "validity_shift",
        "input_sensitivity",
    ),
    "C": (
        # standard counterfactual quality
        "validity",
        "proximity_l1",
        "proximity_l2",
        "sparsity",
        "frac_altered",
        "sparsity_channels",
        "sparsity_timepoints",
        "ood",
        "scm_noise_plausibility",
        "trsi",
        # mechanism faithfulness — the admission gate and its diagnostics
        "cf_faith_rollout_hard",
        "cf_faith_rollout_soft",
        "cf_faith_pearl_hard",
        "cf_faith_pearl_soft",
        "cf_faith_rollout_hard_valid",
        "cf_faith_pearl_hard_valid",
        "n_cf_faith_scorable",
        "frac_degenerate",
        "n_vacuous",
        "frac_vacuous",
        # model-vs-world audit
        "A_model_proposed",
        "B_model_oracle",
        "C_world_oracle",
        "delta_total",
        "delta_trajectory",
        "delta_outcome",
        "do_complexity_mean",
        "do_complexity_median",
    ),
}

#: Reverse index: metric key → axis letter.
AXIS_OF: dict[str, str] = {
    metric: axis for axis, metrics in AXIS_METRICS.items() for metric in metrics
}


def axis_of(metric: str) -> str:
    """Return the axis letter owning ``metric``.

    Raises ``KeyError`` rather than returning a default: a metric with no axis
    is exactly the condition this module exists to prevent, so it must fail
    loudly at the point of use.
    """
    try:
        return AXIS_OF[metric]
    except KeyError:
        raise KeyError(
            f"metric {metric!r} belongs to no axis. Every reported metric must be "
            f"assigned in AXIS_METRICS (causaltemp_xai/metrics/taxonomy.py); see "
            f"DECISIONS.md 2026-08-03."
        ) from None
