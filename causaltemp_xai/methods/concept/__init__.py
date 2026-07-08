"""Concept-based explanation methods.

  CBMT  — Temporal Concept Bottleneck Model (linear probes per causal channel)
  iVAE  — Identifiable VAE for latent disentanglement (Axis A scoring)
"""

from .cbm_t import CBMT
from .ivae import iVAE

__all__ = ["CBMT", "iVAE"]
