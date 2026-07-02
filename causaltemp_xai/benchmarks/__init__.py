"""Benchmark data generation package for causaltemp_xai.

Renamed from benchmark/ to benchmarks/ to mirror causal_tscf_bench structure.
Contains:
  - base.py          : BenchmarkSplit dataclass and BenchmarkDataset ABC
  - linear_scm_t.py  : LinearSCMT and NlinearSCMT generators (MOVED from benchmark/generator.py)
  - mechanisms.py    : LinearMechanism, MLPMechanism, lag_window (MOVED from benchmark/mechanisms.py)
  - structural_cf.py : structural_counterfactual (MOVED from benchmark/structural_cf.py)
"""

from .generator import LinearSCMT, NlinearSCMT
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
    "LinearMechanism",
    "MLPMechanism",
    "Mechanism",
    "lag_window",
    "mechanism_from_state_dict",
]
