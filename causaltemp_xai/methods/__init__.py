"""Counterfactual and attribution methods.

The top-level namespace re-exports the most commonly used classes.
The full method collections are available in the subpackages:
  - causaltemp_xai.methods.counterfactual  (WachterCF, DiCECF, CARLARecourse,
                                             PearlCARLARecourse, cfts_*)
  - causaltemp_xai.methods.attribution     (integrated_gradients, deletion_curve, ...)
"""

from .base import CFExplainer, AttributionMethod
from .counterfactual.wachter import WachterCF
from .counterfactual.dice import DiCECF
from .counterfactual.carla import CARLARecourse, PearlCARLARecourse
from .counterfactual.cfts_methods import (
    CftsWachterCF,
    CftsNativeGuideCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsCelsCF,
)
from .attribution.integrated_gradients import integrated_gradients
from .attribution.perturbation_curves import deletion_curve, insertion_curve

# Backward-compat: derive_intervention_t moved to scm.intervention
from causaltemp_xai.scm.intervention import derive_intervention_t

__all__ = [
    "CFExplainer",
    "AttributionMethod",
    "WachterCF",
    "DiCECF",
    "CARLARecourse",
    "PearlCARLARecourse",
    "CftsWachterCF",
    "CftsNativeGuideCF",
    "CftsCOMTECF",
    "CftsConfetiCF",
    "CftsCountsCF",
    "CftsCelsCF",
    "integrated_gradients",
    "deletion_curve",
    "insertion_curve",
    "derive_intervention_t",
]
