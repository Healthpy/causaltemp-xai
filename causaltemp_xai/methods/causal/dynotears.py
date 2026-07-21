"""DYNOTEARS — Structure Learning from Time-Series Data (Pamfil et al., 2020).

Reference / upstream
--------------------
Pamfil, Sriwattanaworachai, Desai, Pilgerstorfer, Georgatzis, Beaumont, Aragam
(2020). "DYNOTEARS: Structure Learning from Time-Series Data." AISTATS 2020.
Official code: McKinsey CausalNex —
https://github.com/mckinsey/causalnex/blob/develop/causalnex/structure/dynotears.py
**vendored** at ``third_party/causalnex_repo`` (git submodule).

This is a thin **adapter**: the DYNOTEARS solver itself — the NOTEARS augmented-
Lagrangian optimisation over the intra-slice ``W`` and inter-slice (lagged)
``A`` weight matrices with the acyclicity constraint ``h(W)=0`` — is the
upstream ``causalnex.structure.dynotears._learn_dynamic_structure`` used
**unmodified**. The adapter only (i) reshapes the benchmark's ``(N, T, k)``
panel into the ``(X, Xlags)`` DYNOTEARS format (respecting sequence
boundaries), (ii) standardises columns, and (iii) maps the returned lagged
weight matrix ``A`` onto the benchmark's ``(k, k, L)`` adjacency convention.

Why DYNOTEARS here
------------------
DYNOTEARS is a *self-graphing* baseline that actually recovers the lag-1
structure of this benchmark's near-linear SCM (unlike a causal-representation
learner such as CITRIS, whose engine needs nonlinear mixing to have anything to
identify — see ``citris.py``). It gives Axis B's graph-error decomposition real
dynamic range and provides H3 a genuine, non-circular graph-aware method. It is
**observational** — it needs no intervention targets, only the plain dataset.

The class exposes the same ``inferred_graph(max_lag, threshold)`` interface as
:class:`causaltemp_xai.methods.causal.CITRIS`, so the Axis-B pipeline
(``experiments/07_auxiliary_methods.py``) can consume either interchangeably.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np

_CAUSALNEX_REPO = (
    Path(__file__).resolve().parents[3] / "third_party" / "causalnex_repo"
)


def _load_upstream_dynotears():
    """Import the genuine upstream ``_learn_dynamic_structure`` solver.

    ``causalnex/structure/dynotears.py`` imports ``StructureModel`` and
    ``DynamicDataTransformer`` at module top (triggering the heavy causalnex
    package ``__init__``), but the core solver ``_learn_dynamic_structure``
    (numpy + scipy only) uses neither. We load the file directly via importlib
    with those two symbols stubbed, so the genuine optimiser is available
    without importing the rest of causalnex.
    """
    if not _CAUSALNEX_REPO.exists():
        raise ImportError(
            f"vendored causalnex repo not found at {_CAUSALNEX_REPO}. Initialise "
            "the submodule: `git submodule update --init third_party/causalnex_repo`"
        )
    path = _CAUSALNEX_REPO / "causalnex" / "structure" / "dynotears.py"

    # Stub the two causalnex imports dynotears.py makes at module scope (only
    # used by from_numpy_dynamic's StructureModel packaging, which we bypass).
    for name in ("causalnex", "causalnex.structure", "causalnex.structure.transformers"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    sys.modules["causalnex.structure"].StructureModel = object
    sys.modules["causalnex.structure.transformers"].DynamicDataTransformer = object

    spec = importlib.util.spec_from_file_location("causalnex._dynotears_vendored", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_UPSTREAM = None  # lazily loaded


class DYNOTEARS:
    """Fit DYNOTEARS on ``(N, T, k)`` panel data; expose the inferred graph.

    Usage
    -----
    >>> model = DYNOTEARS(k=k, p=L).fit(data["X_train"])
    >>> adj_pred, scores = model.inferred_graph(max_lag=L)

    Parameters
    ----------
    k:
        Number of variables / channels.
    p:
        Number of lag orders to learn (should match the benchmark ``L``).
    lambda_w, lambda_a:
        L1 regularisation on the intra-slice ``W`` / inter-slice ``A`` weights.
    max_iter:
        Max dual-ascent (augmented-Lagrangian) steps.
    w_threshold:
        Absolute-weight threshold applied by the solver (0 keeps raw weights;
        binarisation is left to the caller / density-matching).
    standardize:
        Whether to z-score each channel before fitting (recommended for NOTEARS).
    """

    def __init__(
        self,
        k: int,
        p: int = 1,
        lambda_w: float = 0.01,
        lambda_a: float = 0.01,
        max_iter: int = 100,
        w_threshold: float = 0.0,
        standardize: bool = True,
    ) -> None:
        global _UPSTREAM
        if _UPSTREAM is None:
            _UPSTREAM = _load_upstream_dynotears()
        self.k = k
        self.p = p
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.standardize = standardize
        self.w_est_: Optional[np.ndarray] = None
        self.a_est_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _build_panel(self, X: np.ndarray):
        """Reshape ``(N, T, k)`` -> DYNOTEARS ``(X_now, Xlags)`` respecting
        sequence boundaries (no lag window crosses from one sequence into
        another). ``Xlags`` columns are ``[lag1 | lag2 | ... | lag p]``."""
        X = np.asarray(X, dtype=float)
        N, T, _ = X.shape
        p = self.p
        now_rows, lag_rows = [], []
        for n in range(N):
            for t in range(p, T):
                now_rows.append(X[n, t, :])
                lag_rows.append(np.concatenate([X[n, t - l, :] for l in range(1, p + 1)]))
        return np.asarray(now_rows), np.asarray(lag_rows)

    def fit(self, X: np.ndarray) -> DYNOTEARS:
        """Fit on plain (observational) ``(N, T, k)`` data."""
        X = np.asarray(X, dtype=float)
        if X.shape[2] != self.k:
            raise ValueError(f"X has {X.shape[2]} channels, expected k={self.k}")
        X_now, Xlags = self._build_panel(X)

        if self.standardize:
            mu = X_now.mean(axis=0, keepdims=True)
            sd = X_now.std(axis=0, keepdims=True) + 1e-8
            X_now = (X_now - mu) / sd
            # Apply the *same* per-variable scaling to each lag block of Xlags.
            Xlags = (Xlags - np.tile(mu, self.p)) / np.tile(sd, self.p)

        d = self.k
        # Box constraints exactly as upstream from_numpy_dynamic builds them:
        # ban W self-loops; W/A weights split into non-negative plus/minus.
        bnds_w = 2 * [
            (0, 0) if i == j else (0, None)
            for i in range(d) for j in range(d)
        ]
        bnds_a = []
        for _ in range(self.p):
            bnds_a.extend(2 * [(0, None) for _ in range(d) for _ in range(d)])
        bnds = bnds_w + bnds_a

        w_est, a_est = _UPSTREAM._learn_dynamic_structure(
            X_now, Xlags, bnds, self.lambda_w, self.lambda_a, self.max_iter, 1e-8
        )
        w_est[np.abs(w_est) < self.w_threshold] = 0
        a_est[np.abs(a_est) < self.w_threshold] = 0
        self.w_est_ = w_est          # (d, d) intra-slice (≈0 for lag-only SCM)
        self.a_est_ = a_est          # (p*d, d) inter-slice: [lag*d + from, to]
        return self

    # ------------------------------------------------------------------
    def inferred_graph(
        self, max_lag: int = 1, threshold: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map the learned inter-slice weights ``A`` to ``(k, k, max_lag)``.

        Upstream ``a_est`` has shape ``(p*k, k)`` with ``a_est[l*k + f, t]`` =
        weight of edge (var ``f`` at lag ``l+1``) -> (var ``t`` now). The
        benchmark convention is ``adj[i, j, l]`` == edge ``j -> i`` at lag
        ``l+1``, so ``scores[i, j, l] = |a_est[l*k + j, i]|``. Self-loops are
        zeroed (ground-truth excludes them). Scores are max-normalised to
        ``[0, 1]``; ``adj`` thresholds them.
        """
        if self.a_est_ is None:
            raise RuntimeError("DYNOTEARS is not fitted; call .fit(X) first.")
        k, p = self.k, self.p
        L = max_lag
        scores = np.zeros((k, k, L))
        for l in range(min(p, L)):
            block = np.abs(self.a_est_[l * k : (l + 1) * k, :])  # (k, k): [from, to]
            for i in range(k):
                for j in range(k):
                    scores[i, j, l] = block[j, i]  # j -> i
        for l in range(L):
            np.fill_diagonal(scores[:, :, l], 0.0)
        smax = scores.max()
        if smax > 0:
            scores = scores / smax
        adj = (scores > threshold).astype(int)
        return adj, scores
