"""Attribution / saliency methods.

MOVED from top-level attribution/ package into methods/attribution/.
Added TimeSHAP and Dynamask ported from causal_tscf_bench.
"""

from .integrated_gradients import integrated_gradients
from .perturbation_curves import deletion_curve, insertion_curve
from .timeshap import TimeSHAP
from .dynamask import Dynamask

__all__ = [
    "integrated_gradients",
    "deletion_curve",
    "insertion_curve",
    "TimeSHAP",
    "Dynamask",
]
