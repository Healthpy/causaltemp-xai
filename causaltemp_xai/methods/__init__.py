"""Counterfactual and causal methods.

The top-level namespace re-exports the most commonly used classes.
The full method collections are available in the subpackages:
  - causaltemp_xai.methods.counterfactual  (CARLARecourse, PearlCARLARecourse,
                                             CausalFeasibilityCF, cfts_*)
  - causaltemp_xai.methods.causal          (CITRIS)

The native from-scratch ``WachterCF`` was removed 2026-08-05 (`DECISIONS.md`):
``CftsWachterCF`` wraps the genuine vendored ``cfts`` implementation and was
already what the pipeline used, so the from-scratch reimplementation was
redundant duplication, not a distinct method.
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
    "derive_intervention_t",
]
