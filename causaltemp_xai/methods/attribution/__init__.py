"""Attribution / saliency methods.

MOVED from top-level attribution/ package into methods/attribution/.
"""

from .integrated_gradients import integrated_gradients
from .perturbation_curves import deletion_curve, insertion_curve

__all__ = ["integrated_gradients", "deletion_curve", "insertion_curve"]
