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

# Single source of truth for the lag-window contract (see mechanisms module
# docstring). Re-exported as the module-private ``_window`` for callers/tests.
from causaltemp_xai.benchmarks.mechanisms import lag_window as _window


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
        A :class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism`.

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
        A :class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism`.
    t0:
        Intervention timestep. The prefix ``x_cf[:t0]`` is held to the factual.
    node:
        Variable index intervened on at ``t0``. Either a scalar (single-node
        intervention) or a sequence of indices for a simultaneous multi-node
        intervention — real CF methods routinely edit several channels at
        ``t0``, and scoring only one of them would evaluate a *different*
        intervention than the method actually proposed.
    value:
        The value(s) forced onto ``x_cf[t0, node]``. Must match ``node``'s
        shape.
    noiseless:
        If ``False`` (default), reuse the abducted factual noise — the textbook
        Pearl counterfactual (``pearl_delta``-faithful). If ``True``, roll
        forward with ``eps = 0`` — the deterministic skeleton CF
        (``noiseless_rollout``-faithful).

    Returns
    -------
    ndarray of shape ``(T, k)`` — the oracle counterfactual trajectory.
    """
    return structural_counterfactual_schedule(
        x_orig, mechanism, [(t0, node, value)], noiseless=noiseless
    )


def structural_counterfactual_schedule(
    x_orig: np.ndarray,
    mechanism,
    schedule,
    noiseless: bool = False,
) -> np.ndarray:
    """Oracle counterfactual for a **multi-timestep** intervention schedule.

    Generalises :func:`structural_counterfactual` from a single ``do()`` at
    ``t0`` to a set of them at arbitrary timesteps. Abduction and prediction are
    unchanged — the only difference is that scheduled coordinates are re-imposed
    *after* each mechanism step rather than only seeded at the first one.

    **Why this exists (RISK-18).** The single-``t0`` form is the right object for
    the oracle positive control, where the benchmark chooses the intervention.
    It is the wrong object for auditing a *method's* proposal: real CF methods
    edit many timesteps (CftsWachter alters 98% of all ``T x k`` cells), and
    reading only the first changed slice scores them against an intervention
    they never proposed. See
    :func:`~causaltemp_xai.metrics.pns.extract_intervention_schedule`, which
    recovers the schedule a proposal implies, and the do-complexity metric built
    on it.

    The prefix before the earliest scheduled timestep is held to the factual, as
    in the single-``t0`` case.

    Parameters
    ----------
    x_orig:
        Factual trajectory, shape ``(T, k)``.
    mechanism:
        A :class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism`.
    schedule:
        Non-empty sequence of ``(t, nodes, values)``. ``nodes``/``values`` follow
        :func:`structural_counterfactual`'s contract (matching scalars or
        matching sequences). Timesteps need not be sorted, but must be distinct —
        two entries for one ``t`` are a caller bug, not a merge request.
    noiseless:
        As in :func:`structural_counterfactual`.

    Returns
    -------
    ndarray of shape ``(T, k)`` — the oracle counterfactual trajectory.
    """
    x_orig = np.asarray(x_orig, dtype=float)
    T, k = x_orig.shape
    L = mechanism.L

    by_t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for entry in schedule:
        t, node, value = entry
        t = int(t)
        # ``node``/``value`` are either both scalars (single-node intervention,
        # the original signature) or matching sequences (multi-node). numpy's
        # fancy indexing handles both identically once node is an array, so there
        # is one code path rather than a branch.
        nodes = np.atleast_1d(np.asarray(node, dtype=int))
        values = np.atleast_1d(np.asarray(value, dtype=float))
        if nodes.shape != values.shape:
            raise ValueError(
                f"node and value must have the same shape; got {nodes.shape} and {values.shape}"
            )
        if not 0 <= t < T:
            raise ValueError(f"scheduled timestep {t} out of range for T={T}")
        if t in by_t:
            raise ValueError(f"schedule has two entries for timestep {t}")
        by_t[t] = (nodes, values)

    if not by_t:
        raise ValueError("schedule is empty; an intervention must set at least one value")

    eps = np.zeros((T, k)) if noiseless else abduct_noise(x_orig, mechanism)

    # Action: hold the pre-intervention prefix, then impose the earliest do().
    t_first = min(by_t)
    x_cf = x_orig.copy()
    nodes, values = by_t[t_first]
    x_cf[t_first, nodes] = values

    # Predict: roll forward reusing eps (zero in the noiseless variant),
    # re-imposing any later scheduled values on top of the mechanism's output.
    for t in range(t_first + 1, T):
        x_cf[t] = mechanism.forward_numpy(_window(x_cf, t, L, k)) + eps[t]
        entry = by_t.get(t)
        if entry is not None:
            nodes, values = entry
            x_cf[t, nodes] = values
    return x_cf
