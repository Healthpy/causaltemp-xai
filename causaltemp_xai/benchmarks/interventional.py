"""Intervention-target-labeled temporal sequences (CITRIS data support).

CITRIS (Lippe et al., ICML 2022) learns temporal causal representations from
**temporal intervened sequences**: trajectories in which, at every timestep, a
known random subset of the causal variables is intervened on, and the
per-variable intervention target ``I_t^i in {0, 1}`` is recorded as
supervision. The identifiability result *requires* these target labels — the
observational ``(X, Y, graph, mechanism)`` benchmark artifact does not carry
them, so this module adds the missing data support.

Because CausalTemp-XAI uses **identity mixing** (the ``k`` observed channels
*are* the causal variables — nonlinear mixing ``x = g(z)`` is explicitly out
of scope, see ``generator.py``), the intervention target over causal variables
is simply a mask over channels. Interventions are *perfect* / *stochastic*
(``do(X_t^i = v)`` with ``v`` drawn from an intervention value distribution),
matching the CITRIS training regime.

The rollout reuses the dataset's own ground-truth
:class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism`, so the intervened
sequences are drawn from the *true* SCM: at each step the non-intervened
channels follow ``mechanism.forward_numpy(window) + eps``, while the
intervened channels are overwritten by their sampled ``do`` value (severing
their incoming edges at that step, exactly as the do-operator prescribes).
Downstream channels then propagate the intervention forward through the
mechanism, so the labels are causally grounded, not cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class InterventionalDataset:
    """Container for a batch of intervention-labeled temporal sequences.

    Attributes
    ----------
    X:
        ``(N, T, k)`` observed trajectories under the applied interventions.
    targets:
        ``(N, T, k)`` binary intervention-target masks; ``targets[n, t, i] ==
        1`` iff channel ``i`` was intervened on (``do``-set) at time ``t`` in
        sequence ``n``. By convention ``targets[:, 0, :] == 0`` (the first
        observed step is never intervened, it seeds the transition).
    values:
        ``(N, T, k)`` the ``do`` values that were set where ``targets == 1``
        (``0`` elsewhere); retained so an exact reconstruction of the applied
        intervention is available for evaluation.
    """

    X: np.ndarray
    targets: np.ndarray
    values: np.ndarray

    def as_dict(self) -> dict:
        return {"X": self.X, "targets": self.targets, "values": self.values}


def _default_noise_fn(rng: np.random.Generator, noise_type: str) -> Callable[..., np.ndarray]:
    """Innovation sampler matching the generators' ``_sample_noise`` scales."""

    def sample(*shape) -> np.ndarray:
        size = shape if len(shape) > 1 else shape[0]
        if noise_type == "laplace":
            return rng.laplace(loc=0.0, scale=0.1, size=size)
        if noise_type == "uniform":
            return rng.uniform(low=-0.17, high=0.17, size=size)
        if noise_type == "gaussian":
            return rng.normal(loc=0.0, scale=0.1 * float(np.sqrt(2.0)), size=size)
        raise ValueError(f"unknown noise_type {noise_type!r}")

    return sample


def generate_interventional_sequences(
    mechanism,
    k: int,
    L: int,
    T: int,
    N: int,
    *,
    seed: Optional[int] = 0,
    noise_type: str = "laplace",
    intervention_prob: float = 0.3,
    intervention_scale: float = 1.0,
    burn_in: int = 100,
    clip: float = 1e3,
) -> InterventionalDataset:
    """Roll ``N`` intervention-labeled sequences from a known ``mechanism``.

    At each *observed* timestep ``t >= 1`` and channel ``i``, an intervention
    is applied independently with probability ``intervention_prob``. Where
    applied, ``X_t^i`` is ``do``-set to a value drawn from
    ``Normal(0, intervention_scale)`` (its incoming edges severed for that
    step); elsewhere ``X_t^i = mechanism.forward_numpy(window)_i + eps``. The
    burn-in is intervention-free so every sequence starts from a settled
    factual state.

    Parameters
    ----------
    mechanism:
        A :class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism` (typically
        the dataset's ground-truth ``mechanism``) exposing ``forward_numpy``.
    k, L, T, N:
        Channels, max lag, observed horizon, number of sequences.
    seed:
        RNG seed (interventions, ``do`` values, and noise all derive from it).
    noise_type:
        Innovation distribution — matches the generator families.
    intervention_prob:
        Per-channel, per-step Bernoulli probability of intervening.
    intervention_scale:
        Std-dev of the Normal the ``do`` values are drawn from.
    burn_in:
        Intervention-free warm-up steps discarded from the returned window.
    clip:
        Divergence guard; trajectories are clipped to ``[-clip, clip]``.

    Returns
    -------
    InterventionalDataset
        ``X``/``targets``/``values`` each shaped ``(N, T, k)``.
    """
    if not (0.0 <= intervention_prob <= 1.0):
        raise ValueError(f"intervention_prob must be in [0, 1], got {intervention_prob!r}")
    rng = np.random.default_rng(seed)
    noise_fn = _default_noise_fn(rng, noise_type)

    total_T = T + burn_in
    X = np.zeros((N, total_T, k))
    tgt = np.zeros((N, total_T, k), dtype=np.int8)
    val = np.zeros((N, total_T, k))

    # Seed the first L steps with small noise (as the generators do).
    for lag in range(L):
        X[:, lag, :] = noise_fn(N, k) * 0.1

    noise = noise_fn(N * total_T * k).reshape(N, total_T, k)
    for t in range(L, total_T):
        window = X[:, t - L : t, :]  # (N, L, k)
        mean = mechanism.forward_numpy(window)  # (N, k)
        x_t = mean + noise[:, t, :]
        if t >= burn_in + 1:  # interventions only in the observed window, t>=1
            mask = rng.random((N, k)) < intervention_prob  # (N, k) bool
            do_vals = rng.normal(0.0, intervention_scale, size=(N, k))
            x_t = np.where(mask, do_vals, x_t)
            tgt[:, t, :] = mask.astype(np.int8)
            val[:, t, :] = np.where(mask, do_vals, 0.0)
        X[:, t, :] = x_t

    np.clip(X, -clip, clip, out=X)

    obs = slice(burn_in, None)
    return InterventionalDataset(
        X=X[:, obs, :],
        targets=tgt[:, obs, :],
        values=val[:, obs, :],
    )
