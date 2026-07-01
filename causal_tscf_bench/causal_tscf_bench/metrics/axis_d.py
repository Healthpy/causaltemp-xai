"""
Axis D — Robustness Metrics.

Metrics:
  Shift-VR   : Validity Retention under distribution shift (label-preserving
               covariate shift simulated as Gaussian noise injection)
  InputSens  : Mean sensitivity of explanation to small input perturbations
               (L2 distance between attribution maps under ε-perturbation)
  ConceptStab: Variance of concept activations across semantically equivalent
               instances (low variance = stable concepts)

Reference:
  Alvarez-Melis & Jaakkola (2018), On the Robustness of Interpretability Methods.
  Hsieh et al. (2021), Evaluations and Methods for Explanation through Robustness Analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def shift_vr(X_test: np.ndarray, classifier, cf_explainer,
             target_class: int, sigma: float = 0.1,
             n_trials: int = 5, rng: np.random.Generator | None = None) -> float:
    """
    Shift Validity Retention.

    Inject Gaussian noise (σ = sigma × std(X_test)) into test instances and
    re-run the CF explainer. Measure fraction of shifted instances where the
    CF still achieves target_class.

    Parameters
    ----------
    X_test       : (N, T, M)
    classifier   : TSClassifier with .predict()
    cf_explainer : CFExplainer with .explain()
    target_class : int
    sigma        : noise scale (relative to std(X_test))
    n_trials     : number of noise injections per instance
    rng          : numpy RNG
    """
    if rng is None:
        rng = np.random.default_rng()

    N = X_test.shape[0]
    noise_std = X_test.std() * sigma
    valid_count = 0
    total = 0

    for i in range(N):
        for _ in range(n_trials):
            x_noisy = X_test[i] + rng.normal(0, noise_std, X_test[i].shape)
            try:
                x_cf = cf_explainer.explain(x_noisy, target_class, classifier)
                pred = classifier.predict(x_cf[np.newaxis])[0]
                if int(pred) == target_class:
                    valid_count += 1
            except Exception:
                pass
            total += 1

    return float(valid_count / max(total, 1))


def input_sensitivity(X_test: np.ndarray, attribution_fn: Callable,
                      eps: float = 0.01,
                      n_trials: int = 10,
                      rng: np.random.Generator | None = None) -> float:
    """
    Input Sensitivity of Attribution.

    For each instance, compute L2 distance between attribution under
    original input and under ε-perturbed input (random Gaussian).

    Lower = more stable attributions.

    Parameters
    ----------
    X_test       : (N, T, M)
    attribution_fn: callable(x: (T,M)) -> (T,M) attribution map
    eps          : perturbation scale (absolute)
    n_trials     : perturbations per instance
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
    """
    Concept Stability.

    For a group of semantically equivalent instances (same ground-truth class
    and same causal structure), measure variance of concept activations.
    Low variance = stable / consistent concepts.

    Parameters
    ----------
    concept_fn : callable(x: (T,M)) -> (M,) concept activation per channel
    X_group    : (N, T, M) semantically equivalent instances
    """
    N = X_group.shape[0]
    activations = []
    for i in range(N):
        act = concept_fn(X_group[i])
        activations.append(act.flatten())
    activations = np.array(activations)   # (N, M)
    return float(activations.var(axis=0).mean())


def compute_axis_d(X_test: np.ndarray, classifier,
                   cf_explainer=None,
                   attribution_fn: Callable | None = None,
                   concept_fn: Callable | None = None,
                   target_class: int = 1,
                   sigma: float = 0.1,
                   n_trials: int = 5,
                   rng: np.random.Generator | None = None) -> dict[str, float]:
    """
    Aggregate Axis D metrics.

    All metrics are optional; pass None to skip.
    """
    if rng is None:
        rng = np.random.default_rng()

    results: dict[str, float] = {}

    if cf_explainer is not None:
        results["Shift_VR"] = shift_vr(
            X_test, classifier, cf_explainer, target_class, sigma, n_trials, rng
        )

    if attribution_fn is not None:
        results["InputSens"] = input_sensitivity(X_test, attribution_fn, rng=rng)

    if concept_fn is not None:
        results["ConceptStab"] = concept_stability(concept_fn, X_test)

    return results
