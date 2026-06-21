"""Oracle structural counterfactuals for temporal SCMs.

This module produces the **ground-truth** counterfactual for an atomic
intervention ``do(x[t0, node] = value)`` on a known additive-noise SCM, via
Pearl's abduction-action-prediction recipe:

1. **Abduct** the exogenous noise from the *factual* trajectory. Under additive
   noise (``x_t = mechanism.forward_numpy(window) + eps_t``) this is the exact
   subtraction ``eps[t] = x_orig[t] - mechanism.forward_numpy(window)`` (Hoyer
   ANM 2008; Nasr-Esfahany bijective SCM, ICML 2023).
2. **Action**: hold the pre-intervention prefix ``x_orig[:t0]`` unchanged, copy
   the intervened step ``x_cf[t0] = x_orig[t0]`` and override the single
   coordinate ``x_cf[t0, node] = value``.
3. **Predict**: roll forward ``x_cf[t] = mechanism.forward_numpy(window) +
   eps[t]`` for ``t > t0``, reusing the abducted factual noise.

The result is the canonical **positive control** for
:class:`~causaltemp_xai.metrics.cf_faith.CFfaith`: scoring an oracle CF under
``pearl_delta`` semantics must give ``hard = 1`` by construction. A *noiseless*
variant (``eps = 0``) yields the deterministic skeleton CF that is faithful
under ``noiseless_rollout`` instead — the two are mutually exclusive, which is
exactly the rollout-vs-pearl distinction the benchmark exploits.
"""

from __future__ import annotations

import numpy as np


def _window(arr: np.ndarray, t: int, L: int, k: int) -> np.ndarray:
    """Build the ``(L, k)`` lag window feeding the mechanism at time ``t``.

    Rows ordered oldest→newest; rows reaching before ``t=0`` stay zero (the
    ``if lag_t >= 0`` guard). Mirrors the window contract used everywhere else.
    """
    window = np.zeros((L, k))
    for j in range(L):
        src = t - L + j
        if src >= 0:
            window[j] = arr[src]
    return window


def abduct_noise(x_orig: np.ndarray, mechanism) -> np.ndarray:
    """Recover the exogenous noise ``eps[t] = x_orig[t] - f(window_t)``.

    Exact under additive noise. For ``t < L`` the window is partially (or fully)
    zero-padded, so ``eps[t]`` absorbs the SCM's initial conditions — re-adding
    it to the mechanism rollout reconstructs ``x_orig`` exactly.

    Parameters
    ----------
    x_orig:
        Factual trajectory, shape ``(T, k)``.
    mechanism:
        A :class:`~causaltemp_xai.benchmark.mechanisms.Mechanism`.

    Returns
    -------
    ndarray of shape ``(T, k)`` — the abducted exogenous noise.
    """
    x_orig = np.asarray(x_orig, dtype=float)
    T, k = x_orig.shape
    L = mechanism.L
    eps = np.zeros((T, k))
    for t in range(T):
        eps[t] = x_orig[t] - mechanism.forward_numpy(_window(x_orig, t, L, k))
    return eps


def structural_counterfactual(
    x_orig: np.ndarray,
    mechanism,
    t0: int,
    node: int,
    value: float,
    noiseless: bool = False,
) -> np.ndarray:
    """Ground-truth counterfactual for ``do(x[t0, node] = value)``.

    Implements abduction-action-prediction on the known SCM.

    Parameters
    ----------
    x_orig:
        Factual trajectory, shape ``(T, k)``.
    mechanism:
        A :class:`~causaltemp_xai.benchmark.mechanisms.Mechanism`.
    t0:
        Intervention timestep. The prefix ``x_cf[:t0]`` is held to the factual.
    node:
        Variable index intervened on at ``t0``.
    value:
        The value forced onto ``x_cf[t0, node]``.
    noiseless:
        If ``False`` (default), reuse the abducted factual noise — the textbook
        Pearl counterfactual (``pearl_delta``-faithful). If ``True``, roll
        forward with ``eps = 0`` — the deterministic skeleton CF
        (``noiseless_rollout``-faithful).

    Returns
    -------
    ndarray of shape ``(T, k)`` — the oracle counterfactual trajectory.
    """
    x_orig = np.asarray(x_orig, dtype=float)
    T, k = x_orig.shape
    L = mechanism.L

    eps = np.zeros((T, k)) if noiseless else abduct_noise(x_orig, mechanism)

    # Action: hold the pre-intervention prefix + the intervened step.
    x_cf = x_orig.copy()
    x_cf[t0] = x_orig[t0]
    x_cf[t0, node] = value

    # Predict: roll forward reusing eps (zero in the noiseless variant).
    for t in range(t0 + 1, T):
        x_cf[t] = mechanism.forward_numpy(_window(x_cf, t, L, k)) + eps[t]
    return x_cf
