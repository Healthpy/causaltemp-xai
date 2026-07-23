"""Integrated Gradients (Sundararajan et al., 2017) for the LSTM classifier.

A WP3 *attribution foil*: a faithful saliency method whose maps can score well
on deletion/insertion curves while telling us nothing about causal
faithfulness — the contrast that motivates the benchmark.

Reference
---------
Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep
Networks.* ICML.

Hand-rolled (no ``captum`` dependency) on top of
:meth:`~causaltemp_xai.classifiers.LSTMClassifier.torch_logits`, attributing the
**target-class logit** (a scalar). The default baseline is the all-zeros series
(documented choice — the natural reference for the zero-mean VAR data). The
Riemann sum uses the midpoint rule, which satisfies the IG *completeness* axiom
(``sum(IG) ≈ f_target(x) − f_target(baseline)``) closely for modest step counts.
"""

from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(
    model,
    x: np.ndarray,
    target_class: int,
    baseline: np.ndarray | None = None,
    steps: int = 50,
) -> np.ndarray:
    """Integrated-Gradients attribution map for a single instance.

    Parameters
    ----------
    model:
        Classifier exposing the differentiable ``torch_logits`` hook (accepts a
        ``(T, k)`` tensor, returns ``(1, n_classes)`` logits).
    x:
        Original instance, shape ``(T, k)``.
    target_class:
        Class whose logit is attributed.
    baseline:
        Reference instance, shape ``(T, k)``. Defaults to all zeros.
    steps:
        Number of Riemann-sum (midpoint) interpolation points.

    Returns
    -------
    ndarray of shape ``(T, k)``
        Per-cell attribution. By the completeness axiom the entries sum to
        approximately ``f_target(x) − f_target(baseline)``.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if baseline is None:
        baseline = np.zeros_like(x_arr)
    base_arr = np.asarray(baseline, dtype=np.float32)
    if base_arr.shape != x_arr.shape:
        raise ValueError(f"baseline shape {base_arr.shape} != x shape {x_arr.shape}")

    x_t = torch.as_tensor(x_arr)
    base_t = torch.as_tensor(base_arr)

    # Midpoint Riemann sum: alphas at (j + 0.5) / steps.
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    grad_sum = torch.zeros_like(x_t)
    for a in alphas:
        interp = (base_t + a * (x_t - base_t)).clone().detach().requires_grad_(True)
        logits = model.torch_logits(interp)  # (1, n_classes)
        target_logit = logits[0, target_class]
        (grad,) = torch.autograd.grad(target_logit, interp)
        grad_sum += grad

    avg_grad = grad_sum / steps
    ig = (x_t - base_t) * avg_grad
    return ig.detach().cpu().numpy().astype(np.float32)
