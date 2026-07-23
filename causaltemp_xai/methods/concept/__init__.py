"""Concept-based explanation methods.

ChannelConceptProbe — per-channel logistic concept probe over summary
                      statistics (see module docstring: renamed from
                      ``CBMT`` — it is not a concept bottleneck model;
                      R3 remediation, docs/method_provenance.md)
iVAE                — Identifiable VAE for latent disentanglement (Axis A scoring)
"""

from .channel_concept_probe import ChannelConceptProbe
from .ivae import iVAE

__all__ = ["ChannelConceptProbe", "iVAE"]
