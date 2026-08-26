"""Empirical dynamical diagnostics for SCM mechanisms (M4c).

M4c's DoD explicitly says not to *assume* a contraction rate ``rho ~= 1`` for
the new non-dissipative families (``SpringMechanism``, ``KuramotoMechanism``)
-- it must be **measured**. :func:`empirical_contraction_rate` does that:
paired deterministic rollouts from a tiny initial perturbation, tracking how
the perturbation's norm evolves relative to its start. A negative rate means
the mechanism contracts perturbations (dissipative, like ``MLPMechanism``); a
rate near zero means it roughly conserves them (the non-dissipative target);
a positive rate means it amplifies them.

This is deliberately mechanism-generic (only calls ``mechanism.forward_numpy``
and reads ``mechanism.k``/``.L``), so it applies to any :class:`~causaltemp_xai.
benchmarks.mechanisms.Mechanism` subclass, not just the two new M4c families.
"""

from __future__ import annotations

import numpy as np


def empirical_contraction_rate(
    mechanism,
    T: int = 100,
    n_pairs: int = 50,
    perturbation: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Measure how a mechanism's own deterministic dynamics treat a small
    initial perturbation, averaged over ``n_pairs`` random starting windows.

    For each pair: draw a small random initial ``(L, k)`` window (matching
    the generators' own ``* 0.1`` init-noise convention), perturb one
    randomly chosen channel by ``perturbation``, then roll **both** forward
    for ``T`` steps using only ``mechanism.forward_numpy`` -- no stochastic
    noise added at any step, so the trajectory divergence measured is
    attributable to the mechanism's own dynamics alone, not to independent
    noise draws on the two trajectories.

    Returns
    -------
    dict with keys:

    ``"rate"``
        Least-squares slope of ``log(||delta_t|| / ||delta_0||)`` against
        ``t``, averaged over pairs whose perturbation never vanished to
        exactly zero. Negative = contracting, ~0 = conserved, positive =
        growing.
    ``"ratio_curve"``
        ``(T,)`` array, the pair-averaged ``||delta_t|| / ||delta_0||`` at
        each step (``t=0`` is exactly 1.0 by construction).
    ``"n_pairs"``, ``"T"``
        As passed in.
    """
    rng = np.random.default_rng(seed)
    k, L = mechanism.k, mechanism.L
    ratios = np.zeros((n_pairs, T + 1))

    for p in range(n_pairs):
        window = rng.normal(size=(L, k)) * 0.1
        pert_window = window.copy()
        chan = rng.integers(0, k)
        pert_window[-1, chan] += perturbation

        delta0 = pert_window - window
        norm0 = float(np.linalg.norm(delta0))
        ratios[p, 0] = 1.0  # by construction

        x, xp = window.copy(), pert_window.copy()
        for t in range(T):
            nxt = mechanism.forward_numpy(x[-L:])
            nxt_p = mechanism.forward_numpy(xp[-L:])
            x = np.vstack([x, nxt[None, :]])
            xp = np.vstack([xp, nxt_p[None, :]])
            delta_t = xp[-1] - x[-1]
            ratios[p, t + 1] = float(np.linalg.norm(delta_t)) / norm0

    # Fit log(ratio) vs t; drop any pair/step where ratio hit exactly 0
    # (would be -inf in log) rather than silently biasing the average.
    t_values = np.arange(T + 1)
    mean_ratio = ratios.mean(axis=0)
    valid = mean_ratio > 0
    if valid.sum() >= 2:
        slope, _ = np.polyfit(t_values[valid], np.log(mean_ratio[valid]), 1)
        rate = float(slope)
    else:  # pragma: no cover - defensive, would need every pair to vanish exactly
        rate = float("nan")

    return {
        "rate": rate,
        "ratio_curve": mean_ratio,
        "n_pairs": n_pairs,
        "T": T,
    }
