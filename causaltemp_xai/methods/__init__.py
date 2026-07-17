"""Counterfactual and attribution methods.

The top-level namespace re-exports the most commonly used classes.
The full method collections are available in the subpackages:
  - causaltemp_xai.methods.counterfactual  (WachterCF, CARLARecourse,
                                             PearlCARLARecourse, cfts_*)
  - causaltemp_xai.methods.attribution     (integrated_gradients, TimeSHAP, Dynamask, ...)
  - causaltemp_xai.methods.concept         (CBMT, iVAE)
  - causaltemp_xai.methods.causal          (CITRIS)
"""

# Backward-compat: derive_intervention_t moved to scm.intervention
from causaltemp_xai.scm.intervention import derive_intervention_t

from .attribution.dynamask import Dynamask
from .attribution.integrated_gradients import integrated_gradients
from .attribution.perturbation_curves import deletion_curve, insertion_curve
from .attribution.timeshap import TimeSHAP
from .base import AttributionMethod, CFExplainer
from .causal.citris import CITRIS
from .concept.cbm_t import CBMT
from .concept.ivae import iVAE
from .counterfactual.carla import CARLARecourse, PearlCARLARecourse
from .counterfactual.cfts_methods import (
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsNativeGuideCF,
    CftsWachterCF,
)
from .counterfactual.wachter import WachterCF

__all__ = [
    "CBMT",
    "CITRIS",
    "AttributionMethod",
    "CARLARecourse",
    "CFExplainer",
    "CftsCOMTECF",
    "CftsCelsCF",
    "CftsConfetiCF",
    "CftsCountsCF",
    "CftsNativeGuideCF",
    "CftsWachterCF",
    "Dynamask",
    "PearlCARLARecourse",
    "TimeSHAP",
    "WachterCF",
    "deletion_curve",
    "derive_intervention_t",
    "iVAE",
    "insertion_curve",
    "integrated_gradients",
]
