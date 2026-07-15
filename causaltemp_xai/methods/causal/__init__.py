"""Causal / graph-aware methods (self-graphing baselines for Axis B / H3).

  DYNOTEARS — Structure Learning from Time-Series Data (Pamfil et al., 2020;
              genuine McKinsey CausalNex solver, vendored). Classical temporal
              causal-discovery baseline that recovers the benchmark's lag-1
              structure from OBSERVATIONAL data (AUC ~0.9). The load-bearing
              graph-aware method: gives Axis B's graph-error decomposition real
              dynamic range and H3 a genuine, non-circular positive.

  CITRIS — Causal Identifiability from Temporal Intervened Sequences
           (Lippe et al., ICML 2022; genuine upstream modules, vendored).
           Representation learner; needs intervention-target-labeled data.
           An honest *secondary* method — its identifiability engine needs
           nonlinear mixing to have anything to identify, so on this
           identity-mixing benchmark it does not recover the graph at smoke
           scale (see its class docstring).

Note on iCITRIS: iCITRIS (Lippe et al., 2022) extends CITRIS with a causal
discovery mechanism for *instantaneous* (lag-0) effects. CausalTemp-XAI's SCMs
are purely time-lagged (graph over lags >= 1, no instantaneous edges), so
iCITRIS's instantaneous-discovery component reduces to plain CITRIS on this
benchmark and is deliberately not implemented as a redundant second model —
see the class docstring of :class:`CITRIS`.
"""

from .citris import CITRIS
from .dynotears import DYNOTEARS

__all__ = ["CITRIS", "DYNOTEARS"]
