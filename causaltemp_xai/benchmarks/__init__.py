"""Benchmark data generation package for causaltemp_xai.

Renamed from benchmark/ to benchmarks/ to mirror causal_tscf_bench structure.
Contains:
  - generator.py     : LinearSCMT and NlinearSCMT generators (dict-returning generate())
  - mechanisms.py    : LinearMechanism, MLPMechanism, lag_window
  - structural_cf.py : structural_counterfactual
"""

from .generator import (
    HMMRegimeSwitchNlinearSCMT,
    LinearSCMT,
    NlinearSCMT,
    RegimeSwitchNlinearSCMT,
)
from .mechanisms import (
    LinearMechanism,
    MLPMechanism,
    Mechanism,
    lag_window,
    mechanism_from_state_dict,
)

__all__ = [
    "LinearSCMT",
    "NlinearSCMT",
    "RegimeSwitchNlinearSCMT",
    "HMMRegimeSwitchNlinearSCMT",
    "LinearMechanism",
    "MLPMechanism",
    "Mechanism",
    "lag_window",
    "mechanism_from_state_dict",
]
