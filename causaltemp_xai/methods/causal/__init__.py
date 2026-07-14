"""Causal / graph-aware representation-learning methods.

  CITRIS — Causal Identifiability from Temporal Intervened Sequences
           (Lippe et al., ICML 2022). Learns temporal causal variables from
           intervention-target-labeled sequences and exposes an inferred
           lagged causal graph for Axis B graph-error decomposition.

Note on iCITRIS: iCITRIS (Lippe et al., 2022) extends CITRIS with a causal
discovery mechanism for *instantaneous* (lag-0) effects. CausalTemp-XAI's SCMs
are purely time-lagged (graph over lags >= 1, no instantaneous edges), so
iCITRIS's instantaneous-discovery component reduces to plain CITRIS on this
benchmark and is deliberately not implemented as a redundant second model —
see the class docstring of :class:`CITRIS`.
"""

from .citris import CITRIS

__all__ = ["CITRIS"]
