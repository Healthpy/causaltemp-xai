"""Wachter et al. (2017) gradient-based counterfactual method — stub.

Reference
---------
Wachter, S., Mittelstadt, B., & Russell, C. (2017).
*Counterfactual explanations without opening the black box: Automated
decisions and the GDPR.*  Harvard Journal of Law & Technology, 31(2).

The method minimises the loss

    L(cf) = lambda * loss_pred(f(cf), target) + dist(cf, x)

with respect to ``cf``, where:

* ``loss_pred`` penalises predictions that differ from the target class,
* ``dist`` is the L2 distance between the counterfactual and the original,
* ``lambda`` trades off prediction fidelity against proximity.

This implementation operates on **PyTorch** tensors to enable automatic
differentiation through differentiable classifiers.  A numpy wrapper is
provided for convenience.
"""

from __future__ import annotations

import numpy as np


class WachterCF:
    """Stub: Wachter counterfactual generator.

    The real implementation should minimise the following loss w.r.t. ``cf``
    using gradient descent through a differentiable model:

        L(cf) = lambda * yloss(f(cf), y_target) + ||cf - x||_2^2

    where ``yloss`` is typically hinge loss on the output logit for
    ``y_target``, and ``lambda`` is increased (or ``cf`` re-initialised) when
    no valid CF is found within ``n_steps`` optimisation iterations.

    Parameters
    ----------
    target_class : int
        Desired output class.
    lam : float
        Regularisation weight (trades off prediction loss vs proximity).
    lr : float
        Adam learning rate for the CF variable.
    n_steps : int
        Maximum gradient-descent steps per instance.
    tol : float
        Early-stopping threshold on the prediction loss.
    """

    def __init__(
        self,
        target_class: int = 1,
        lam: float = 0.1,
        lr: float = 0.01,
        n_steps: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.target_class = target_class
        self.lam = lam
        self.lr = lr
        self.n_steps = n_steps
        self.tol = tol

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate a single counterfactual for instance *x*.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)`` or ``(k,)``
            Original time-series instance.
        model : callable
            Differentiable classifier; must accept a batch tensor ``(N, k, T)``
            and return logits of shape ``(N, n_classes)``.

        Returns
        -------
        cf : ndarray, same shape as ``x``.

        Implementation notes
        --------------------
        1. Wrap ``x`` as a leaf ``torch.Tensor`` with ``requires_grad=False``.
        2. Initialise ``cf`` as a copy of ``x`` with ``requires_grad=True``.
        3. Run Adam on ``cf`` for up to ``n_steps`` steps minimising
           ``L(cf) = lam * cross_entropy(model(cf), target) + ||cf-x||^2``.
        4. Stop early once the model predicts ``target_class`` (pred loss < tol).
        5. Return ``cf.detach().numpy()``.
        """
        raise NotImplementedError(
            "WachterCF.generate() is a stub. Implement gradient-based CF search."
        )
