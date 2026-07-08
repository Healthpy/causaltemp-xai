"""Attribution / saliency methods.

MOVED from top-level attribution/ package into methods/attribution/.

``FDSaliency`` (finite-difference saliency) and ``MCMaskSHAP`` (Monte-Carlo
masking SHAP approximation), ported from causal_tscf_bench, are disclosed
proxies for Dynamask and TimeSHAP respectively -- see their module
docstrings (``fd_saliency.py``, ``mc_mask_shap.py``) and
``docs/method_provenance.md`` for what they actually compute and why they
are not named after those papers. These classes were named ``Dynamask`` and
``TimeSHAP`` in an earlier revision of this codebase; that naming has been
retired (docs/PROJECT_PLAN.md Standing Decision #3, risk R4).
"""

from .integrated_gradients import integrated_gradients
from .perturbation_curves import deletion_curve, insertion_curve
from .mc_mask_shap import MCMaskSHAP
from .fd_saliency import FDSaliency

__all__ = [
    "integrated_gradients",
    "deletion_curve",
    "insertion_curve",
    "MCMaskSHAP",
    "FDSaliency",
]
