"""Counterfactual and causal methods.

The top-level namespace re-exports the most commonly used classes.
The full method collections are available in the subpackages:
  - causaltemp_xai.methods.counterfactual  (NoiselessSCMRecourse, PearlSCMRecourse,
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
from .counterfactual.causal_feasibility import TSCausalCF
from .counterfactual.cfts_methods import (
    CftsCelsCF,
    CftsCOMTECF,
    CftsConfetiCF,
    CftsCountsCF,
    CftsNativeGuideCF,
    CftsWachterCF,
)
from .counterfactual.scm_recourse import NoiselessSCMRecourse, PearlSCMRecourse

__all__ = [
    "AttributionMethod",
    "CFExplainer",
    "CftsCOMTECF",
    "CftsCelsCF",
    "CftsConfetiCF",
    "CftsCountsCF",
    "CftsNativeGuideCF",
    "CftsWachterCF",
    "NoiselessSCMRecourse",
    "PCMCIPlus",
    "PearlSCMRecourse",
    "TSCausalCF",
    "derive_intervention_t",
]
