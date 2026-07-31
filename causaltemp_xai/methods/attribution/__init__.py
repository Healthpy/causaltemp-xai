"""Attribution / saliency methods.

MOVED from top-level attribution/ package into methods/attribution/.

Both ``TimeSHAP`` and ``Dynamask`` are now the genuine methods, replacing the
earlier disclosed proxies:

- ``TimeSHAP`` wraps the official ``timeshap`` library (Bento et al., 2021),
  replacing the ``MCMaskSHAP`` flat Monte-Carlo random-coalition sampler.
- ``Dynamask`` wraps the authors' official Dynamask implementation, vendored as
  a git submodule at ``third_party/dynamask_repo/`` (Crabbe & van der Schaar,
  2021), replacing the ``FDSaliency`` finite-difference proxy.

Both proxies had been renamed off their original method names under
R3 (``docs/risk_register.md`` RISK-04); shipping the real methods
resolves that concern.
"""

from .dynamask import Dynamask
from .integrated_gradients import integrated_gradients
from .perturbation_curves import deletion_curve, insertion_curve
from .timeshap import TimeSHAP

__all__ = [
    "Dynamask",
    "TimeSHAP",
    "deletion_curve",
    "insertion_curve",
    "integrated_gradients",
]
