"""
Lagged causal DAG sampling for TSCM benchmarks.

Ported from causal_tscf_bench/causal_tscf_bench/scm/dag.py.

A LaggedDAG encodes which lagged variables X_{t-tau}^(i) causally drive X_t^(j).
Adjacency shape: (n_channels, n_channels, max_lag) where
  adjacency[j, i, tau-1] = 1  means  X_{t-tau}^(i) -> X_t^(j).
Instantaneous edges (tau=0) are allowed only when they form a DAG among channels
(enforced by topological ordering). Lagged edges (tau>=1) never create cycles.

Variable naming: n_channels = k (number of variables/channels), consistent with
causaltemp_xai's (N, T, k) convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LaggedDAG:
    n_channels: int
    max_lag: int
    # shape (n_channels, n_channels, max_lag)
    # adjacency[j, i, lag_idx] == 1  =>  X_{t-(lag_idx+1)}^(i) -> X_t^(j)
    adjacency: np.ndarray
    # topological order over channels (for instantaneous edges, not used in lag>=1 case)
    topo_order: list[int] = field(default_factory=list)

    def parents_of(self, j: int) -> list[tuple[int, int]]:
        """Return list of (channel_i, lag) pairs that are parents of channel j."""
        parents = []
        for i in range(self.n_channels):
            for lag_idx in range(self.max_lag):
                if self.adjacency[j, i, lag_idx] == 1:
                    parents.append((i, lag_idx + 1))
        return parents

    @property
    def n_edges(self) -> int:
        return int(self.adjacency.sum())

    @property
    def edge_density(self) -> float:
        max_possible = self.n_channels**2 * self.max_lag
        return self.n_edges / max_possible if max_possible > 0 else 0.0


def sample_dag(
    n_channels: int,
    max_lag: int,
    edge_density: float,
    rng: np.random.Generator,
    allow_self_loops: bool = True,
) -> LaggedDAG:
    """
    Sample a random time-lagged causal DAG.

    For lagged edges (tau >= 1) cycles are impossible by temporal precedence,
    so we draw each potential edge independently with probability = edge_density.

    Parameters
    ----------
    n_channels : int
        Number of observed channels k.
    max_lag : int
        Maximum causal lag tau_max.
    edge_density : float
        Bernoulli probability for each candidate edge.
    rng : np.random.Generator
    allow_self_loops : bool
        Whether X_{t-tau}^(j) -> X_t^(j) (auto-regressive) edges are allowed.

    Returns
    -------
    LaggedDAG
    """
    adjacency = np.zeros((n_channels, n_channels, max_lag), dtype=np.int8)

    for lag_idx in range(max_lag):
        for j in range(n_channels):
            for i in range(n_channels):
                if not allow_self_loops and i == j:
                    continue
                if rng.random() < edge_density:
                    adjacency[j, i, lag_idx] = 1

    # Ensure at least one edge exists to avoid degenerate trivial benchmarks
    if adjacency.sum() == 0:
        j = rng.integers(n_channels)
        i = rng.integers(n_channels)
        adjacency[j, i, 0] = 1

    return LaggedDAG(
        n_channels=n_channels,
        max_lag=max_lag,
        adjacency=adjacency,
        topo_order=list(range(n_channels)),
    )
