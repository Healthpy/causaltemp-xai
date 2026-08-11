"""Counterfactual and causal methods.

The top-level namespace re-exports the most commonly used classes.
The full method collections are available in the subpackages:
  - causaltemp_xai.methods.counterfactual  (CARLARecourse, PearlCARLARecourse,
                                             TSCausalCF, cfts_*)
  - causaltemp_xai.methods.causal          (DYNOTEARS, PCMCIPlus)

The native from-scratch ``WachterCF`` was removed 2026-08-05:
``CftsWachterCF`` wraps the genuine vendored ``cfts`` implementation and was
already what the pipeline used, so the from-scratch reimplementation was
redundant duplication, not a distinct method.
"""

# Backward-compat: derive_intervention_t moved to scm.intervention
from causaltemp_xai.scm.intervention import derive_intervention_t

from .base import AttributionMethod, CFExplainer
from .causal.pcmci import PCMCIPlus
from .counterfactual.carla import CARLARecourse, PearlCARLARecourse
from .counterfactual.causal_feasibility import TSCausalCF
from .counterfactual.cfts_methods import (
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsNativeGuideCF,
    CftsWachterCF,
)

__all__ = [
    "AttributionMethod",
    "CARLARecourse",
    "CFExplainer",
    "CftsCOMTECF",
    "CftsCelsCF",
    "CftsConfetiCF",
    "CftsCountsCF",
    "CftsNativeGuideCF",
    "CftsWachterCF",
    "PCMCIPlus",
    "PearlCARLARecourse",
    "TSCausalCF",
    "derive_intervention_t",
]
