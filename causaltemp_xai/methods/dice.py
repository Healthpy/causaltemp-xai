"""DiCE: Diverse Counterfactual Explanations (Mothilal et al., 2020) — stub.

Reference
---------
Mothilal, R. K., Sharma, A., & Tan, C. (2020).
*Explaining machine learning classifiers through diverse counterfactual
explanations.*  Proceedings of the 2020 ACM FAccT.

DiCE generates a *set* of diverse counterfactuals by jointly optimising:

    L = proximity_loss + diversity_loss + prediction_loss

* **proximity_loss** – L2 distance from each CF to the original.
* **diversity_loss** – determinantal point process (DPP) term that
  encourages the CFs to be spread out in feature space.
* **prediction_loss** – cross-entropy between the CF prediction and the
  target class.
"""

from __future__ import annotations

import numpy as np


class DiCECF:
    """Stub: DiCE diverse counterfactual generator.

    The real implementation should jointly optimise a set of *n_cfs*
    counterfactuals by minimising:

        L = sum_i pred_loss(cf_i) + lam_prox * sum_i ||cf_i - x||^2
              - lam_div * log det K

    where ``K`` is the pairwise-similarity kernel matrix between CFs
    (determinantal point process diversity term).

    Parameters
    ----------
    target_class : int
        Desired output class.
    n_cfs : int
        Number of diverse CFs to generate per instance.
    lam_prox : float
        Proximity regularisation weight.
    lam_div : float
        DPP diversity weight.
    lr : float
        Adam learning rate.
    n_steps : int
        Optimisation iterations.
    """

    def __init__(
        self,
        target_class: int = 1,
        n_cfs: int = 5,
        lam_prox: float = 0.5,
        lam_div: float = 1.0,
        lr: float = 0.01,
        n_steps: int = 1000,
    ) -> None:
        self.target_class = target_class
        self.n_cfs = n_cfs
        self.lam_prox = lam_prox
        self.lam_div = lam_div
        self.lr = lr
        self.n_steps = n_steps

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate *n_cfs* diverse counterfactuals for a single instance.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)`` or ``(k,)``
            Original time-series instance.
        model : callable
            Differentiable classifier accepting a batch tensor and returning
            logits of shape ``(N, n_classes)``.

        Returns
        -------
        cfs : ndarray of shape ``(n_cfs, T, k)``.

        Implementation notes
        --------------------
        1. Initialise ``n_cfs`` candidate CFs close to ``x`` (e.g. add small
           Gaussian jitter so they are not identical and the DPP term is
           non-degenerate).
        2. Jointly optimise all CFs with a single Adam optimiser.
        3. Compute the DPP kernel ``K[i,j] = 1 / (1 + ||cf_i - cf_j||^2)``
           and use ``-logdet(K)`` as the diversity loss.
        4. Return the optimised CF array.
        """
        raise NotImplementedError(
            "DiCECF.generate() is a stub. Implement joint DPP-diversity CF search."
        )
