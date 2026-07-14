"""
Invertible operator dictionary Phi for TSCM mechanism sampling.

Ported from causal_tscf_bench/causal_tscf_bench/scm/operators.py.

Each operator phi_ij is applied element-wise to a parent channel's lagged value.
All operators in INVERTIBLE_OPERATORS admit exact noise abduction.
The "step" operator is non-invertible and is reserved for adversarial ablations.

Reference: BenchmarkingTSCFEs.md Phase 1 operator dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Full operator pool including the adversarial step function
OPERATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "abs": np.abs,
    "square": lambda x: x ** 2,
    "exp_neg_abs": lambda x: np.exp(-np.abs(x)),
    # Non-invertible: only used in non-monotonic ablation
    "step": lambda x: (x > 0).astype(float),
}

# Operators that allow exact abduction via Plan Eq. 2 (all except step)
INVERTIBLE_OPERATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    k: v for k, v in OPERATORS.items() if k != "step"
}

OPERATOR_KEYS = list(INVERTIBLE_OPERATORS.keys())
ADVERSARIAL_KEYS = list(OPERATORS.keys())


@dataclass
class Mechanism:
    """Per-edge mechanism: a scalar weight and an operator key.

    NOTE: This is the bench-style Mechanism dataclass (lightweight, per-edge).
    It is distinct from causaltemp_xai.benchmarks.mechanisms.Mechanism
    (which is the LinearMechanism/MLPMechanism class used for VAR generation).
    Both coexist: this Mechanism is used in the new scm/ package; the
    LinearMechanism/MLPMechanism is used in benchmarks/ for VAR generation.
    """
    channel_from: int
    channel_to: int
    lag: int
    weight: float
    operator_key: str

    def apply(self, x_lagged: np.ndarray) -> np.ndarray:
        """Compute w_ij * phi_ij(x_{t-tau}^(i))."""
        return self.weight * OPERATORS[self.operator_key](x_lagged)


def sample_mechanism(
    dag: "LaggedDAG",  # noqa: F821 – forward ref
    rng: np.random.Generator,
    sigma_w: float = 1.0,
    operator_pool: dict | None = None,
) -> list[Mechanism]:
    """
    Sample a mechanism (weight + operator) for every edge in the DAG.

    Parameters
    ----------
    dag : LaggedDAG
    rng : np.random.Generator
    sigma_w : float
        Standard deviation for weight sampling w_ij ~ N(0, sigma_w^2).
    operator_pool : dict | None
        Operator pool to sample from. Defaults to INVERTIBLE_OPERATORS.

    Returns
    -------
    list[Mechanism]
        One Mechanism per edge in the DAG.
    """
    if operator_pool is None:
        operator_pool = INVERTIBLE_OPERATORS

    keys = list(operator_pool.keys())
    mechanisms = []

    for j in range(dag.n_channels):
        for i, lag in dag.parents_of(j):
            weight = rng.normal(0.0, sigma_w)
            op_key = rng.choice(keys)
            mechanisms.append(Mechanism(
                channel_from=i,
                channel_to=j,
                lag=lag,
                weight=weight,
                operator_key=op_key,
            ))

    return mechanisms
