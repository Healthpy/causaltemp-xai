"""Counterfactual and causal methods.

The top-level namespace re-exports the most commonly used classes.
The full method collections are available in the subpackages:
  - causaltemp_xai.methods.counterfactual  (WachterCF, CARLARecourse,
                                             PearlCARLARecourse, cfts_*)
  - causaltemp_xai.methods.causal          (CITRIS)
"""

# Backward-compat: derive_intervention_t moved to scm.intervention
from causaltemp_xai.scm.intervention import derive_intervention_t

from .base import AttributionMethod, CFExplainer
from .causal.citris import CITRIS
from .counterfactual.carla import CARLARecourse, PearlCARLARecourse
from .counterfactual.causal_feasibility import CausalFeasibilityCF
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
    "CITRIS",
    "AttributionMethod",
    "CARLARecourse",
    "CFExplainer",
    "CausalFeasibilityCF",
    "CftsCOMTECF",
    "CftsCelsCF",
    "CftsConfetiCF",
    "CftsCountsCF",
    "CftsNativeGuideCF",
    "CftsWachterCF",
    "PearlCARLARecourse",
    "WachterCF",
    "derive_intervention_t",
]
