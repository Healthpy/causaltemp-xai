"""Official Dynamask attribution (Crabbe & van der Schaar, 2021, ICML).

This module wraps the *official* Dynamask implementation by the paper authors,
vendored as a git submodule at ``third_party/dynamask_repo/`` (mirroring
``third_party/cfts_repo/``). It learns a soft temporal mask
``M in [0, 1]^{T x k}`` by gradient-based optimization of Dynamask's
perturbation objective (prediction-preservation vs. a size regulator and a
temporal-smoothness penalty) and returns the fitted mask as the ``(T, k)``
importance map.

This replaces the earlier ``FDSaliency`` proxy -- a per-coordinate
finite-difference numerical gradient that was *not* Dynamask and shipped under
a disclosure note (docs/PROJECT_PLAN.md Standing Decision #3 / risk R4). Because
this is now the genuine method, no proxy disclosure is required.

Reference: Crabbe, J. & van der Schaar, M. (2021). Explaining Time Series
Predictions with Dynamic Masks. ICML 2021. Official implementation:
https://github.com/JonathanCrabbe/Dynamask (vendored submodule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ...classifiers.base import TSClassifier
from ..base import AttributionMethod

# The vendored Dynamask modules use repo-root-relative absolute imports
# (``from attribution.perturbation import ...``, ``from utils.losses import
# ...``), so the submodule root must be importable. Appended (not inserted at
# 0) so it cannot shadow same-named top-level modules elsewhere; the repo's
# ``attribution``/``utils`` dirs do not collide with anything on the path.
_DYNAMASK_ROOT = Path(__file__).parents[3] / "third_party" / "dynamask_repo"
if str(_DYNAMASK_ROOT) not in sys.path:
    sys.path.append(str(_DYNAMASK_ROOT))


class Dynamask(AttributionMethod):
    """Official Dynamask dynamic-mask saliency, as a dense ``(T, k)`` map.

    Dynamask explains the classifier's *predictive distribution as a whole* --
    it optimizes the mask to preserve ``f(x)`` under perturbation -- so unlike
    the other attribution methods it is class-agnostic: ``target_class`` is
    accepted for interface compatibility but does not change the learned mask.

    Parameters
    ----------
    n_epoch : int, default 200
        Number of mask-optimization steps (Dynamask's ``Mask.fit`` default is
        500; 200 is a faster setting adequate for these short sequences).
    keep_ratio : float, default 0.5
        Fraction of ``(t, feature)`` cells the mask should keep (``a`` in the
        paper).
    learning_rate : float, default 0.1
        SGD learning rate for the mask optimizer.
    time_reg_factor : float, default 1.0
        Temporal-smoothness regularization weight (``lambda_a`` in the paper).
    sigma_max : float, default 2
        Maximum Gaussian-blur width for the perturbation operator.
    momentum : float, default 0.9
        SGD momentum for the mask optimizer.
    perturbation : {"gaussian_blur", "fade_moving_average"}, default "gaussian_blur"
        Dynamask perturbation operator applied to masked-out entries.
    seed : int, default 42
        Random seed for reproducibility (threaded into ``Mask.random_seed``).
    verbose : bool, default False
        Print per-epoch optimization metrics.
    """

    def __init__(
        self,
        n_epoch: int = 200,
        keep_ratio: float = 0.5,
        learning_rate: float = 1.0e-1,
        time_reg_factor: float = 1.0,
        sigma_max: float = 2,
        momentum: float = 0.9,
        perturbation: str = "gaussian_blur",
        seed: int = 42,
        verbose: bool = False,
    ):
        self.n_epoch = n_epoch
        self.keep_ratio = keep_ratio
        self.learning_rate = learning_rate
        self.time_reg_factor = time_reg_factor
        self.sigma_max = sigma_max
        self.momentum = momentum
        self.perturbation = perturbation
        self.seed = seed
        self.verbose = verbose

    def attribute(
        self, x: np.ndarray, classifier: TSClassifier, target_class: int | None = None
    ) -> np.ndarray:
        import contextlib
        import io

        import torch
        from attribution.mask import Mask
        from attribution.perturbation import FadeMovingAverage, GaussianBlur
        from utils.losses import log_loss

        dev = getattr(classifier, "device", torch.device("cpu"))
        X = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # (T, k)

        def f(X_in: torch.Tensor) -> torch.Tensor:
            logits = classifier.torch_logits(X_in)  # (1, n_classes)
            return torch.softmax(logits, dim=-1)[0]  # (n_classes,)

        if self.perturbation == "fade_moving_average":
            pert = FadeMovingAverage(dev)
        else:
            pert = GaussianBlur(dev, sigma_max=self.sigma_max)

        mask = Mask(pert, dev, task="classification", verbose=self.verbose, random_seed=self.seed)
        # Mask.fit prints an unconditional completion banner; silence it (and
        # any per-epoch output) unless the caller asked to be verbose.
        sink = (
            contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())
        )
        with sink:
            mask.fit(
                X,
                f,
                loss_function=log_loss,
                keep_ratio=self.keep_ratio,
                n_epoch=self.n_epoch,
                learning_rate=self.learning_rate,
                time_reg_factor=self.time_reg_factor,
                momentum=self.momentum,
            )
        return mask.mask_tensor.detach().cpu().numpy().astype(np.float64)
