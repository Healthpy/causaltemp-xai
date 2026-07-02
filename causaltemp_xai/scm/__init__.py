"""SCM sub-package: lagged DAG, operators, simulation, abduction, and counterfactuals.

Ported from causal_tscf_bench/causal_tscf_bench/scm/ with minor adaptations
to use (N, T, k) variable naming consistent with causaltemp_xai conventions
(k = number of channels/variables; semantically identical to bench's M).
"""

from .dag import LaggedDAG, sample_dag
from .operators import OPERATORS, INVERTIBLE_OPERATORS, Mechanism, sample_mechanism
from .tscm import simulate_tscm, sample_noise
from .abduction import abduct
from .intervention import derive_intervention_t
from .counterfactual import compute_gt_counterfactual

__all__ = [
    "LaggedDAG",
    "sample_dag",
    "OPERATORS",
    "INVERTIBLE_OPERATORS",
    "Mechanism",
    "sample_mechanism",
    "simulate_tscm",
    "sample_noise",
    "abduct",
    "derive_intervention_t",
    "compute_gt_counterfactual",
]
