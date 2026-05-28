"""CARLA-style causal recourse — stub.

Reference
---------
Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter, T., & Kasneci, G.
(2021). *CARLA: A Python Library to Benchmark Algorithmic Recourse and
Counterfactual Explanation Algorithms.*  NeurIPS 2021 Datasets and Benchmarks.

This stub defines the ``CARLARecourse`` class interface.  The real
implementation would train a generative model (e.g. CVAE) or apply
constrained optimisation that respects actionability constraints and the
causal ordering over variables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class CARLARecourse:
    """Stub: CARLA-style causal recourse generator.

    The real implementation should:

    1. Encode the SCM causal ordering to determine which variables are
       *actionable* (can be intervened on by the individual).
    2. Optimise only over the perturbation delta on actionable variables:

           delta_actionable = delta * actionable_mask
           cf = x + delta_actionable

    3. Minimise ``lam_pred * pred_loss(model(cf), target) + lam_prox * ||delta||^2``.
    4. Propagate the intervention through the causal graph to update
       non-actionable downstream variables consistently.

    Alternatively, a CVAE-based approach can be used to sample plausible
    recourses directly from the learned latent space.

    Parameters
    ----------
    target_class : int
        Desired output class.
    actionable_mask : ndarray or None
        Boolean mask of shape ``(T, k)`` or ``(k,)``.  ``True`` marks
        features that may be changed.  ``None`` means all features are
        actionable.
    lam_pred : float
        Prediction-loss weight.
    lam_prox : float
        Proximity regularisation weight.
    lr : float
        Adam learning rate.
    n_steps : int
        Gradient-descent iterations.
    """

    def __init__(
        self,
        target_class: int = 1,
        actionable_mask: Optional[np.ndarray] = None,
        lam_pred: float = 1.0,
        lam_prox: float = 0.5,
        lr: float = 0.01,
        n_steps: int = 1000,
    ) -> None:
        self.target_class = target_class
        self.actionable_mask = actionable_mask
        self.lam_pred = lam_pred
        self.lam_prox = lam_prox
        self.lr = lr
        self.n_steps = n_steps

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate a causally-constrained recourse for instance *x*.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)`` or ``(k,)``
            Original time-series instance.
        model : callable
            Differentiable classifier accepting a batch tensor and returning
            logits of shape ``(N, n_classes)``.

        Returns
        -------
        cf : ndarray, same shape as ``x``.

        Implementation notes
        --------------------
        1. Optimise a perturbation ``delta`` masked to actionable features.
        2. After optimisation, propagate changes through the causal graph for
           non-actionable downstream variables using ``mechanisms``.
        3. Return ``(x + delta_masked).numpy()``.
        """
        raise NotImplementedError(
            "CARLARecourse.generate() is a stub. Implement causal recourse."
        )
