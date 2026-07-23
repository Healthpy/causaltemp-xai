"""Wachter et al. (2017) gradient-based counterfactual method â€” stub.

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
import torch
import torch.nn.functional as F


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

    #: Multiplicative factor and max rounds for lambda escalation when no flip.
    _LAM_GROWTH = 10.0
    _MAX_ROUNDS = 5

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate a single counterfactual for instance *x*.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)``
            Original time-series instance (public ``(T, k)`` layout).
        model : LSTMClassifier
            Differentiable classifier exposing ``torch_logits`` (accepts a
            ``(T, k)``/``(N, T, k)`` tensor, returns ``(N, n_classes)`` logits).

        Returns
        -------
        cf : ndarray of shape ``(T, k)``.

        Notes
        -----
        Minimises ``L(cf) = lam * CE(model.torch_logits(cf), target) + ||cf-x||^2``
        with Adam on a ``cf`` leaf tensor initialised at ``x``; stops early once
        the prediction flips to ``target_class``. If no flip is found within
        ``n_steps``, ``lam`` is escalated (``x10``, up to 5 rounds) to prioritise
        the flip over proximity.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        x_t = torch.as_tensor(x_arr)  # constant reference (T, k)
        target = torch.tensor([self.target_class], dtype=torch.long)

        cf = x_t.clone().detach().requires_grad_(True)
        best_cf = x_t.clone()
        lam = float(self.lam)

        for _round in range(self._MAX_ROUNDS):
            optimiser = torch.optim.Adam([cf], lr=self.lr)
            for _step in range(self.n_steps):
                optimiser.zero_grad()
                logits = model.torch_logits(cf)  # (1, n_classes)
                pred_loss = F.cross_entropy(logits, target)
                prox = ((cf - x_t) ** 2).sum()
                loss = lam * pred_loss + prox
                loss.backward()
                optimiser.step()

                if int(logits.argmax(dim=1).item()) == self.target_class:
                    # Flip achieved â€” record and stop escalating.
                    return cf.detach().cpu().numpy().astype(np.float32)
            # No flip this round: keep the closest attempt, raise lam, continue.
            best_cf = cf.detach().clone()
            lam *= self._LAM_GROWTH

        return best_cf.cpu().numpy().astype(np.float32)

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        X = np.asarray(X, dtype=np.float32)
        return np.stack([self.generate(x, model) for x in X], axis=0)

    # ------------------------------------------------------------------
    # CFExplainer alias interface
    # ------------------------------------------------------------------

    def fit(self, X_train, classifier) -> None:
        """No-op — WachterCF needs no training data."""
        pass

    def explain(self, x, target_class: int, classifier) -> np.ndarray:
        """Alias for generate(x, classifier) with target_class from constructor."""
        return self.generate(x, classifier)
