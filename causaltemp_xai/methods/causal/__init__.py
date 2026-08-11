"""Causal / graph-aware methods (self-graphing baselines for Axis B / H3).

  DYNOTEARS — Structure Learning from Time-Series Data (Pamfil et al., 2020;
              genuine McKinsey CausalNex solver, vendored). Classical temporal
              causal-discovery baseline that recovers the benchmark's lag-1
              structure from OBSERVATIONAL data (AUC ~0.9). The load-bearing
              graph-aware method: gives Axis B's graph-error decomposition real
              dynamic range and H3 a genuine, non-circular positive.

  PCMCIplus — Constraint-based causal discovery for time series (Runge et al.,
              2020; `tigramite` package, installed as a normal PyPI
              dependency, not vendored). Conditional-independence-testing
              approach (`ParCorr`), structurally unlike DYNOTEARS's
              continuous-optimization NOTEARS formulation. Wired 2026-08-06
              (M4h) specifically as the second,
              different-model-class causal-discovery method
              `docs/risk_register.md` RISK-22 names as the mitigation for
              "bootstrap-ensemble variance is not model-misspecification
              bias" — a DYNOTEARS-vs-PCMCIplus cross-method agreement check
              is the concrete instrument. Observational-only (`fit(X)`, same
              shape as DYNOTEARS); has no `to_linear_mechanism()` equivalent
              (`ParCorr`'s test statistics are not rollout-usable regression
              coefficients), so it does not participate in M4g's Tier-2
              discovered-mechanism CF-faith pipeline.

CITRIS (Lippe et al., ICML 2022) was vendored here through 2026-08-06 and was
deleted that day (M4h) along with its vendored submodule and
`benchmarks/interventional.py` — it required intervention-target-labeled data
with no real-data analogue, and its own docstring already conceded near-chance
graph recovery on this identity-mixing synthetic benchmark. PCMCIplus replaces
it as the second self-graphing method.
"""

from .dynotears import DYNOTEARS
from .pcmci import PCMCIPlus

__all__ = ["DYNOTEARS", "PCMCIPlus"]
