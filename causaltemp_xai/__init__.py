"""causaltemp-xai: Causal Counterfactual Explanations for Temporal Data."""

from causaltemp_xai.eval import evaluate_method
from causaltemp_xai.eval import shift_vr as shift_vr_methods
from causaltemp_xai.classifiers.lstm import LSTMClassifier
from causaltemp_xai.methods.counterfactual.wachter import WachterCF
from causaltemp_xai.methods.counterfactual.dice import DiCECF
from causaltemp_xai.methods.counterfactual.carla import CARLARecourse
from causaltemp_xai.metrics import (
    CFfaith,
    validity,
    proximity,
    sparsity,
    ood_plausibility,
    icc,
    icc_latent,
    compute_axis_a,
    compute_axis_b,
    compute_axis_c,
    compute_axis_d,
)
from causaltemp_xai.scm.intervention import derive_intervention_t

__version__ = "0.2.0"

__all__ = [
    # evaluation
    "evaluate_method",
    "shift_vr_methods",
    # classifiers
    "LSTMClassifier",
    # CF methods
    "WachterCF",
    "DiCECF",
    "CARLARecourse",
    # metrics
    "CFfaith",
    "validity",
    "proximity",
    "sparsity",
    "ood_plausibility",
    "icc",
    "icc_latent",
    "compute_axis_a",
    "compute_axis_b",
    "compute_axis_c",
    "compute_axis_d",
    # utils
    "derive_intervention_t",
]
