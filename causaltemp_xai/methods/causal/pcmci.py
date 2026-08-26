"""PCMCIplus — Constraint-Based Causal Discovery for Time Series (Runge, 2020).

Reference / upstream
---------------------
Runge, J. (2020). "Discovering contemporaneous and lagged causal relations
in autocorrelated nonlinear time series datasets." UAI 2020 (PCMCI+).
Package: ``tigramite`` (https://github.com/jakobrunge/tigramite), installed
as a normal PyPI dependency (2026-08-06, M4h) — unlike
DYNOTEARS/CITRIS/``cfts``/``dynamask``, **not** vendored as a git submodule.
This is a deliberate exception, confirmed directly with the user, not an
inconsistency with this repo's usual vendoring convention.

``tigramite`` is ordinary installed numpy/scipy-only Python with no unrelated
heavy import chain to route around — unlike ``dynotears.py``'s
``_load_upstream_dynotears`` trick, which exists solely to reach
``causalnex.structure.dynotears``'s core solver without importing the rest of
the (much heavier) vendored ``causalnex`` package. No equivalent problem
exists here, so this adapter uses plain top-of-file imports; no
``importlib``-level loading indirection is needed.

Why PCMCIplus here
-------------------
Wired 2026-08-06 (M4h) specifically as the second,
structurally different causal-discovery method `docs/risk_register.md`
RISK-22 names as the missing mitigation for "bootstrap-ensemble variance is
not model-misspecification bias" (M4f/M4g). PCMCIplus is a constraint-based /
conditional-independence-testing method (here: ``ParCorr``, linear/Gaussian
partial correlation — matching this benchmark's own largely near-linear VAR
structure, the same spirit as DYNOTEARS's own linear-SEM baseline role),
structurally unlike DYNOTEARS's continuous-optimization NOTEARS formulation.
A DYNOTEARS-vs-PCMCIplus cross-method agreement check (SHD between their
independently inferred graphs) therefore tests whether two differently-biased
model classes agree — a check within-model bootstrap resampling structurally
cannot perform.

Like DYNOTEARS, and unlike CITRIS (deleted 2026-08-06, M4h), PCMCIplus is
**observational**: ``.fit(X)`` takes a single positional array, no
intervention targets.

**Hard limitation, stated up front, not discovered later:** PCMCIplus has no
``to_linear_mechanism()`` equivalent. ``ParCorr``'s test statistics are
partial-correlation *strengths*, not regression coefficients usable for a
rollout mechanism — so PCMCIplus cannot participate in M4g's Tier-2
discovered-*mechanism* CF-faith pipeline
(``experiments/07b_discovered_graph_real.py``), which stays DYNOTEARS-only.
PCMCIplus participates only in graph-level analyses: ``GRAPH_METHODS``, M4e's
graph-quality sweep, and the cross-method agreement check.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


class PCMCIPlus:
    """Fit PCMCIplus on ``(N, T, k)`` panel data; expose the inferred graph.

    Usage
    -----
    >>> model = PCMCIPlus(k=k, tau_max=L).fit(data["X_train"])
    >>> adj_pred, scores = model.inferred_graph(max_lag=L)

    Parameters
    ----------
    k:
        Number of variables / channels.
    tau_max:
        Maximum lag order to test (should match the benchmark ``L``, the same
        role as :class:`~causaltemp_xai.methods.causal.dynotears.DYNOTEARS`'s
        ``p``).
    pc_alpha:
        PCMCIplus's PC-stage significance threshold for edge removal. Has no
        DYNOTEARS equivalent — DYNOTEARS is a continuous L1-regularised
        optimisation with no comparable alpha; this is a genuinely new
        adapter-level knob, not a renamed existing one.
    standardize:
        Whether to z-score each channel before fitting. ``ParCorr`` (partial
        correlation) is scale-invariant for edge *strength*, but
        per-channel standardisation keeps behaviour predictable and
        comparable to DYNOTEARS's own default (``standardize=True``).
    """

    #: Fixed internal constant, not a constructor parameter: excludes
    #: contemporaneous (lag-0) edges, matching this benchmark's own
    #: no-instantaneous-edges convention (see DYNOTEARS's
    #: ``test_no_instantaneous_edges_learned``). This also sidesteps
    #: tau=0's orientation-ambiguous ``'o-o'``/``'x-x'`` marks entirely — at
    #: ``tau >= 1`` a lagged link's direction is unambiguous by time
    #: ordering, so only ``'-->'`` or no-link is possible there (confirmed
    #: empirically, not just from docs: fitting on synthetic lag-1 data never
    #: produced anything but ``'-->'``/``''`` at ``tau=1``). Not exposed as a
    #: constructor parameter: doing so would let a caller silently
    #: reintroduce the ambiguity this design exists to sidestep.
    _TAU_MIN = 1

    def __init__(
        self,
        k: int,
        tau_max: int = 1,
        pc_alpha: float = 0.05,
        standardize: bool = True,
    ) -> None:
        self.k = k
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.standardize = standardize
        self.graph_: Optional[np.ndarray] = None
        self.val_matrix_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> PCMCIPlus:
        """Fit on plain (observational) ``(N, T, k)`` data."""
        X = np.asarray(X, dtype=float)
        if X.shape[2] != self.k:
            raise ValueError(f"X has {X.shape[2]} channels, expected k={self.k}")

        if self.standardize:
            mu = X.mean(axis=(0, 1), keepdims=True)
            sd = X.std(axis=(0, 1), keepdims=True) + 1e-8
            X = (X - mu) / sd

        # tigramite's pp.DataFrame(analysis_mode="multiple") accepts a 3D
        # array shaped (M, T, N) -- M datasets/trajectories, T timesteps, N
        # variables -- which IS this benchmark's own (N_traj, T, k) array
        # positionally: benchmark's N_traj == tigramite's M (trajectories),
        # benchmark's k == tigramite's N (variables). Confirmed empirically
        # against the installed package, not assumed from docs alone -- no
        # transpose or reshape needed, unlike DYNOTEARS's hand-rolled
        # per-sequence panel construction (``_build_panel``).
        df = pp.DataFrame(X, analysis_mode="multiple")
        pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr())
        results = pcmci.run_pcmciplus(
            tau_min=self._TAU_MIN, tau_max=self.tau_max, pc_alpha=self.pc_alpha
        )
        self.graph_ = results["graph"]  # (k, k, tau_max+1) string edge marks
        self.val_matrix_ = results["val_matrix"]  # same shape, float, signed
        return self

    # ------------------------------------------------------------------
    def inferred_graph(
        self, max_lag: int = 1, threshold: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map tigramite's string-typed lagged graph to ``(k, k, max_lag)``.

        tigramite's ``graph_[i, j, tau] == '-->'`` (``tau > 0``) means a
        directed lagged link **i (at t-tau) -> j (at t)** -- first index is
        the SOURCE. This benchmark's convention is ``adj[i, j, l] == 1``
        means edge ``j -> i`` at lag ``l+1`` -- first index is the TARGET.
        So: ``adj[i, j, l] = 1 if graph_[j, i, l+1] == '-->' else 0`` -- note
        both the index swap (``i<->j``) AND the tau shift (this benchmark's
        ``l=0`` is tigramite's ``tau=1``, since ``tau_min=1`` at fit time
        excludes ``tau=0`` entirely). ``scores[i, j, l] = |val_matrix_[j, i,
        l+1]|`` (same swap+shift, using the absolute test-statistic strength
        -- ``ParCorr``'s ``val_matrix`` is signed). Self-loops are zeroed
        (matching :meth:`DYNOTEARS.inferred_graph
        <causaltemp_xai.methods.causal.dynotears.DYNOTEARS.inferred_graph>`'s
        convention). Scores are max-normalised to ``[0, 1]``; ``adj``
        thresholds them.

        If ``max_lag`` exceeds the ``tau_max`` used at :meth:`fit` time, the
        excess lag slices stay zero (mirrors DYNOTEARS's ``min(p, max_lag)``
        behaviour) rather than raising.
        """
        if self.graph_ is None:
            raise RuntimeError("PCMCIPlus is not fitted; call .fit(X) first.")
        k = self.k
        L = max_lag
        fitted_tau_max = self.graph_.shape[2] - 1
        scores = np.zeros((k, k, L))
        for l in range(min(L, fitted_tau_max)):
            tau = l + 1
            for i in range(k):
                for j in range(k):
                    if self.graph_[j, i, tau] == "-->":
                        scores[i, j, l] = abs(self.val_matrix_[j, i, tau])
        for l in range(L):
            np.fill_diagonal(scores[:, :, l], 0.0)
        smax = scores.max()
        if smax > 0:
            scores = scores / smax
        adj = (scores > threshold).astype(int)
        return adj, scores
