"""SCM sub-package: lagged DAG, operators, simulation, abduction, and counterfactuals.

Ported from causal_tscf_bench/causal_tscf_bench/scm/ with minor adaptations
to use (N, T, k) variable naming consistent with causaltemp_xai conventions
(k = number of channels/variables; semantically identical to bench's M).
"""

from .abduction import abduct
from .counterfactual import compute_gt_counterfactual
from .dag import LaggedDAG, sample_dag
from .intervention import INTERVENTION_TOL, derive_intervention_t
from .operators import INVERTIBLE_OPERATORS, OPERATORS, Mechanism, sample_mechanism
from .tscm import sample_noise, simulate_tscm

__all__ = [
    "INTERVENTION_TOL",
    "INVERTIBLE_OPERATORS",
    "OPERATORS",
    "LaggedDAG",
    "Mechanism",
    "abduct",
    "compute_gt_counterfactual",
    "derive_intervention_t",
    "sample_dag",
    "sample_mechanism",
    "sample_noise",
    "simulate_tscm",
]
