"""Axis B - Robustness metrics.

Full port from causal_tscf_bench/metrics/axis_b.py.

Metrics:
  InputSens  : Mean sensitivity of explanation to small input perturbations
  ConceptStab: Variance of concept activations across semantically equivalent instances

Reference:
  Alvarez-Melis & Jaakkola (2018), On the Robustness of Interpretability Methods.
  Hsieh et al. (2021), Evaluations and Methods for Explanation through Robustness Analysis.

NOTE: Shift-VR (validity retention under environment shift) lives in the
top-level ``eval.py`` — the guarded, adversarially tested, base-CF-reusing
protocol the pipeline uses. This module previously carried a second, unused
``shift_vr`` (Gaussian noise-injection variant) under the same name; it was
removed 2026-07-18 (metric-quality fix #3): an untested duplicate that also
silently counted explainer crashes as invalidity (``except: pass``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def input_sensitivity(
    X_test: np.ndarray,
    attribution_fn: Callable,
    eps: float = 0.01,
    n_trials: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """Input Sensitivity of Attribution.

    For each instance, compute L2 distance between attribution under
    original input and under epsilon-perturbed input (random Gaussian).
    Lower = more stable attributions.

    Parameters
    ----------
    X_test       : (N, T, k)
    attribution_fn: callable(x: (T,k)) -> (T,k) attribution map
    eps          : perturbation scale (absolute)
    n_trials     : perturbations per instance

    Returns
    -------
    float -- mean relative L2 sensitivity
    """
    if rng is None:
        rng = np.random.default_rng()

    N = X_test.shape[0]
    sensitivities = []

    for i in range(N):
        phi_orig = attribution_fn(X_test[i]).flatten()
        trial_diffs = []
        for _ in range(n_trials):
            delta = rng.normal(0, eps, X_test[i].shape)
            phi_pert = attribution_fn(X_test[i] + delta).flatten()
            norm_diff = np.linalg.norm(phi_pert - phi_orig) / (np.linalg.norm(phi_orig) + 1e-8)
            trial_diffs.append(norm_diff)
        sensitivities.append(np.mean(trial_diffs))

    return float(np.mean(sensitivities))


def concept_stability(concept_fn: Callable, X_group: np.ndarray) -> float:
    """Concept Stability.

    For a group of semantically equivalent instances (same ground-truth class
    and same causal structure), measure variance of concept activations.
    Low variance = stable / consistent concepts.

    Parameters
    ----------
    concept_fn : callable(x: (T,k)) -> (k,) concept activation per channel
    X_group    : (N, T, k) semantically equivalent instances

    Returns
    -------
    float -- mean variance of concept activations across group
    """
    N = X_group.shape[0]
    activations = []
    for i in range(N):
        act = concept_fn(X_group[i])
        activations.append(np.asarray(act).flatten())
    activations = np.array(activations)  # (N, k)
    return float(activations.var(axis=0).mean())


def compute_axis_b(
    X_test: np.ndarray,
    classifier=None,
    attribution_fn: Callable | None = None,
    concept_fn: Callable | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Aggregate Axis B metrics.

    All metrics are optional; pass None to skip. Shift-VR for CF methods is
    the ``eval.shift_vr`` benchmark-level protocol, not part of this module
    (see module docstring).

    Parameters
    ----------
    X_test       : (N, T, k)
    classifier   : kept for API compatibility; unused
    attribution_fn : callable or None
    concept_fn   : callable or None
    rng          : numpy RNG

    Returns
    -------
    dict with subset of: InputSens, ConceptStab
    """
    if rng is None:
        rng = np.random.default_rng()

    results: dict = {}

    if attribution_fn is not None:
        results["InputSens"] = input_sensitivity(X_test, attribution_fn, rng=rng)

    if concept_fn is not None:
        results["ConceptStab"] = concept_stability(concept_fn, X_test)

    return results
