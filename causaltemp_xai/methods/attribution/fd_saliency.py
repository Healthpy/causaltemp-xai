"""Finite-difference saliency — a numerical-gradient perturbation baseline.

This module implements a simple black-box saliency method: at every (t,
feature) coordinate it estimates the local sensitivity of the classifier's
predicted-class confidence via a forward finite difference,

    phi[t, m] = (P(y=target | x + eps * e_{t,m}) - P(y=target | x)) / eps

This is a standard forward-difference numerical-gradient approximation. It
is NOT an implementation of Crabbe & van der Schaar's Dynamask.

Naming history / disclosure
----------------------------
An earlier revision of this codebase named this class ``Dynamask`` and its
docstring described it as using "finite-difference saliency as a proxy for
the full Dynamask optimization." That naming violated this lab's standing
decision that no proxy implementation ships under an original method's name
(docs/PROJECT_PLAN.md, Standing Decision #3) and is tracked as risk R4
("misrepresented baselines... fatal if published"). The class has been
renamed to ``FDSaliency`` to describe what it actually computes. No
algorithmic change was made in the rename — only the name, docstrings, and
(see below) the constructor signature, which dropped three keyword
arguments (``n_steps``, ``lr``, ``lambda_sparsity``) that were vestiges of
the Dynamask-shaped API this class never actually implemented (there is no
optimization loop here to consume a step count, learning rate, or sparsity
penalty). Neither the old nor the new class was wired into any experiment
script or covered by a test before this change.

What this is NOT (relationship to the real Dynamask)
-----------------------------------------------------
Dynamask (Crabbe & van der Schaar, 2021, "Explaining Time Series Predictions
with Dynamic Masks", ICML 2021) learns a *soft temporal mask*
``m in [0, 1]^{T x k}`` by gradient-based optimization of a perturbation
objective that trades off (a) how much the mask preserves the classifier's
prediction under a learned blurring/baseline perturbation against (b) a
sparsity/entropy penalty on the mask, optionally with an "extremal mask"
formulation and rate-distortion extensions. None of that optimization,
regularization, or learned-perturbation machinery is implemented here.
``FDSaliency`` instead computes a per-coordinate numerical gradient by
direct forward-difference perturbation — closer in spirit to vanilla
gradient saliency / SmoothGrad-style sensitivity maps than to Dynamask.

The reference below is retained only as "the method this class was
originally intended to approximate" — it is not a description of this
class's identity or an implementation source.

Reference (approximation target, NOT an implementation source): Crabbe, J.
& van der Schaar, M. (2021). Explaining Time Series Predictions with Dynamic
Masks. ICML 2021.

See docs/method_provenance.md for the short provenance note (implementation
source, paradigm approximated, what a real integration would require).

Ported from causal_tscf_bench/methods/attribution/dynamask.py (file renamed
from dynamask.py to fd_saliency.py as part of this disclosure).
"""

from __future__ import annotations

import numpy as np

from ..base import AttributionMethod
from ...classifiers.base import TSClassifier


class FDSaliency(AttributionMethod):
    """Finite-difference (numerical-gradient) saliency for time series classifiers.

    For every time step ``t`` and feature ``m``, perturbs ``x[t, m]`` by a
    small ``eps`` and measures the resulting change in the target class's
    predicted probability, divided by ``eps``:

        phi[t, m] = (P(y=target | x + eps * e_{t,m}) - P(y=target | x)) / eps

    This is a first-order forward-difference approximation of
    ``d P(y=target | x) / d x[t, m]``. It only requires ``predict_proba``
    (no autodiff, no access to model internals), so it works with any
    black-box classifier, at the cost of ``T * k`` forward passes per
    explained instance.

    This is NOT Dynamask (Crabbe & van der Schaar, 2021): it does not learn
    a soft mask, does not optimize any perturbation objective, and has no
    sparsity or smoothness regularization. See the module docstring for the
    full disclosure.

    Parameters
    ----------
    eps : float, default 1e-3
        Finite-difference step size added to each perturbed coordinate.
        The only hyperparameter that affects the computed attribution.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def attribute(self, x: np.ndarray, classifier: TSClassifier,
                  target_class: int | None = None) -> np.ndarray:
        """Compute the ``(T, k)`` finite-difference saliency map for ``x``.

        Parameters
        ----------
        x : np.ndarray
            Shape ``(T, k)`` — single factual instance.
        classifier : TSClassifier
            Must expose ``predict`` and ``predict_proba``.
        target_class : int | None
            Class to attribute toward. Defaults to the classifier's
            predicted class for ``x``.

        Returns
        -------
        np.ndarray
            Shape ``(T, k)`` signed finite-difference sensitivity scores.
        """
        T, k = x.shape
        if target_class is None:
            target_class = int(classifier.predict(x[np.newaxis])[0])

        eps = self.eps
        phi = np.zeros((T, k), dtype=np.float64)
        base_conf = classifier.predict_proba(x[np.newaxis])[0, target_class]

        for t in range(T):
            for m in range(k):
                x_pert = x.copy()
                x_pert[t, m] += eps
                conf_pert = classifier.predict_proba(x_pert[np.newaxis])[0, target_class]
                phi[t, m] = (conf_pert - base_conf) / eps

        return phi
