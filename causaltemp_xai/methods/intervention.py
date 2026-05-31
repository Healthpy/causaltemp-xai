"""Intervention-time derivation — the hinge of the CausalTemp-XAI benchmark.

CF methods (Wachter, DiCE) return a fully-perturbed trajectory with no declared
intervention point, but :class:`~causaltemp_xai.metrics.cf_faith.CFfaith` needs
an ``intervention_t``. The benchmark derives it **uniformly** from any CF as the
first timestep at which the CF deviates from the original — applied identically
to every method so CF-faith scores are comparable. See the plan index "crux".
"""

from __future__ import annotations

import numpy as np


def derive_intervention_t(x: np.ndarray, x_cf: np.ndarray, tol: float = 1e-6) -> int:
    """Return the smallest ``t`` such that ``max_j |x_cf[t,j] - x[t,j]| > tol``.

    Parameters
    ----------
    x, x_cf:
        Original and counterfactual trajectories, shape ``(T, k)``.
    tol:
        Per-element threshold below which a timestep is considered unchanged.

    Returns
    -------
    int
        The first changed timestep. If no timestep differs (the CF equals the
        original within ``tol``), returns ``T - 1`` (degenerate; CF == original).
    """
    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    T = x.shape[0]
    # Max absolute deviation per timestep (over the feature axis/axes).
    per_t = np.abs(x_cf - x).reshape(T, -1).max(axis=1)
    changed = np.flatnonzero(per_t > tol)
    if changed.size == 0:
        return T - 1
    return int(changed[0])
