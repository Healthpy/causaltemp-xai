"""causaltemp-xai: Causal Counterfactual Explanations for Temporal Data."""

from causaltemp_xai.classifiers.lstm import LSTMClassifier
from causaltemp_xai.eval import evaluate_method
from causaltemp_xai.eval import shift_vr as shift_vr_methods
from causaltemp_xai.methods.counterfactual.carla import CARLARecourse
from causaltemp_xai.methods.counterfactual.wachter import WachterCF
from causaltemp_xai.metrics import (
    CFfaith,
    compute_axis_a,
    compute_axis_b,
    compute_axis_c,
    compute_axis_d,
    icc,
    icc_latent,
    ood_plausibility,
    proximity,
    sparsity,
    validity,
)
from causaltemp_xai.scm.intervention import derive_intervention_t

__version__ = "0.2.0"

__all__ = [
    "CARLARecourse",
    # metrics
    "CFfaith",
    # classifiers
    "LSTMClassifier",
    # CF methods
    "WachterCF",
    "compute_axis_a",
    "compute_axis_b",
    "compute_axis_c",
    "compute_axis_d",
    # utils
    "derive_intervention_t",
    # evaluation
    "evaluate_method",
    "icc",
    "icc_latent",
    "ood_plausibility",
    "proximity",
    "shift_vr_methods",
    "sparsity",
    "validity",
]
