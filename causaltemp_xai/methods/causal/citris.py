"""CITRIS — Causal Identifiability from Temporal Intervened Sequences.

Reference / upstream
--------------------
Lippe, Magliacane, Löwe, Asano, Cohen, Gavves (2022).
"CITRIS: Causal Identifiability from Temporal Intervened Sequences." ICML 2022.
Official code: https://github.com/phlippe/CITRIS — **vendored** at
``third_party/citris_repo`` (git submodule).

This module is a thin **adapter**, not a reimplementation: the identifiability-
critical machinery — the intervention-conditioned transition prior with a
Gumbel-Softmax latent-to-causal-variable assignment ``psi`` (including the
``psi(0)`` intervention-independent slot) — is the upstream
``models.shared.transition_prior.TransitionPrior`` used **unmodified**. The
adapter only supplies the observation-modality encoder/decoder (an MLP over the
``k``-dim time-series channels, in place of the repo's image CNNs — the modality
encoder is legitimately swappable, and the repo itself ships several) and a
training loop that optimises the CITRIS-VAE ELBO
(reconstruction + ``TransitionPrior.kl_divergence``).

Data
----
CITRIS trains on temporal *intervened* sequences with per-step intervention
targets ``I^{t+1}`` (which causal variable was intervened). Generate them with
:func:`causaltemp_xai.benchmarks.interventional.generate_interventional_sequences`
using ``mode="single"`` (at most one intervention per step → one-hot / all-zero
targets, the standard CITRIS regime).

Adaptation notes (honest scope)
-------------------------------
* **Identity mixing.** The benchmark's ``k`` channels *are* the causal
  variables (nonlinear mixing ``x = g(z)`` is out of scope), so ``num_blocks =
  k`` and the MLP encoder/decoder are shallow. CITRIS's representation-
  identifiability result (recovering causal variables under *unknown* nonlinear
  mixing) is therefore not stress-tested by this benchmark — what is exercised
  is the genuine transition prior + Gumbel-Softmax target assignment.
* **Inferred lagged graph.** CITRIS-VAE does not emit a causal graph directly;
  :meth:`inferred_graph` derives a lag-1 adjacency from the trained transition
  prior's input-sensitivity, aggregated over causal blocks via the learned
  ``psi`` assignment (see the method docstring).
* **iCITRIS** (instantaneous effects) is not wired: the benchmark's SCM is
  purely time-lagged (no lag-0 edges), so iCITRIS reduces to CITRIS here.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Repo root -> third_party/citris_repo
_CITRIS_REPO = (
    Path(__file__).resolve().parents[3] / "third_party" / "citris_repo"
)


def _load_upstream_modules():
    """Import the genuine upstream ``TransitionPrior`` and ``TargetClassifier``.

    The upstream package ``__init__`` files pull in PyTorch-Lightning /
    torchvision (image-pipeline deps not installed here), so we load the four
    plain-``torch`` source files we need directly via importlib, registering
    lightweight stubs for the heavy optional deps and namespace packages so the
    intra-repo ``from models.shared.X import Y`` imports resolve **without**
    executing the Lightning-dependent package ``__init__``.
    """
    if not _CITRIS_REPO.exists():
        raise ImportError(
            f"vendored CITRIS repo not found at {_CITRIS_REPO}. Initialise the "
            "submodule: `git submodule update --init third_party/citris_repo`"
        )
    shared = _CITRIS_REPO / "models" / "shared"

    # Stub heavy optional deps only if genuinely missing (their symbols are not
    # touched by the transition prior / target classifier / modules / the two
    # pure-torch util functions we use).
    for dep in ("torchvision", "seaborn", "matplotlib", "matplotlib.pyplot"):
        if dep not in sys.modules:
            try:
                __import__(dep)
            except Exception:
                sys.modules[dep] = types.ModuleType(dep)

    # Namespace packages so `models.shared.X` resolves without running __init__.
    for name in ("models", "models.shared"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # mark as package
            sys.modules[name] = pkg

    def _load(mod_name: str, path: Path):
        if mod_name in sys.modules and getattr(sys.modules[mod_name], "__file__", None):
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Order matters: modules + utils first (transition_prior imports from them).
    _load("models.shared.modules", shared / "modules.py")
    _load("models.shared.utils", shared / "utils.py")
    tp = _load("models.shared.transition_prior", shared / "transition_prior.py")
    tc = _load("models.shared.target_classifier", shared / "target_classifier.py")
    return tp.TransitionPrior, tc.TargetClassifier


TransitionPrior = None  # lazily loaded on first CITRIS construction
TargetClassifier = None


class _CITRISVAE(nn.Module):
    """MLP encoder/decoder + the genuine upstream ``TransitionPrior``.

    Latents are grouped into ``num_blocks = k`` causal blocks (plus the prior's
    internal ``psi(0)`` noise slot); ``num_latents = k * latents_per_block``.
    """

    def __init__(self, k: int, latents_per_block: int = 2, c_hid: int = 32,
                 lambda_reg: float = 0.01):
        super().__init__()
        self.k = k
        self.num_blocks = k
        self.num_latents = k * latents_per_block

        self.enc = nn.Sequential(
            nn.Linear(k, c_hid), nn.SiLU(),
            nn.Linear(c_hid, 2 * self.num_latents),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.num_latents, c_hid), nn.SiLU(),
            nn.Linear(c_hid, k),
        )
        self.prior = TransitionPrior(
            num_latents=self.num_latents,
            num_blocks=self.num_blocks,
            c_hid=c_hid,
            imperfect_interventions=False,
            autoregressive_model=False,
            lambda_reg=lambda_reg,
        )
        # Genuine upstream target classifier — the auxiliary loss that
        # specialises the Gumbel-Softmax assignment psi (latents -> causal
        # variables). Without it, psi stays near-uniform and the recovered
        # graph is uninformative.
        self.intv_classifier = TargetClassifier(
            num_latents=self.num_latents,
            c_hid=c_hid,
            num_blocks=self.num_blocks,
        )

    def encode(self, x: torch.Tensor):
        """``x`` ``(B, k)`` -> ``(mu, logstd)`` each ``(B, num_latents)``."""
        h = self.enc(x)
        mu, logstd = h.chunk(2, dim=-1)
        return mu, logstd

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    @staticmethod
    def _reparam(mu, logstd):
        return mu + torch.randn_like(mu) * logstd.exp()


class CITRIS:
    """Fit genuine CITRIS-VAE on intervention-labeled sequences.

    Usage
    -----
    >>> from causaltemp_xai.benchmarks.interventional import (
    ...     generate_interventional_sequences)
    >>> ds = generate_interventional_sequences(mechanism, k, L, T, N, mode="single")
    >>> model = CITRIS(k=k).fit(ds.X, ds.targets)
    >>> adj_pred, scores = model.inferred_graph(max_lag=L)

    Parameters
    ----------
    k:
        Number of causal variables / channels (``num_blocks``).
    latents_per_block:
        Latent dimensions per causal block; ``num_latents = k * latents_per_block``.
    c_hid:
        Hidden width of the encoder/decoder and the transition-prior network.
    beta:
        KL weight (β-VAE style) on the transition-prior KL.
    lambda_reg:
        Upstream ``TransitionPrior`` regulariser mass on the ``psi(0)`` noise slot.
    lr, max_epochs, batch_size:
        Optimisation hyperparameters.
    seed:
        RNG seed for weight init + reparameterisation + Gumbel sampling.
    device:
        Torch device string; defaults to CPU.
    """

    def __init__(
        self,
        k: int,
        latents_per_block: int = 2,
        c_hid: int = 32,
        beta: float = 1.0,
        beta_classifier: float = 2.0,
        lambda_reg: float = 0.01,
        gumbel_max: float = 2.0,
        gumbel_min: float = 0.5,
        lr: float = 1e-3,
        max_epochs: int = 30,
        batch_size: int = 128,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        global TransitionPrior, TargetClassifier
        if TransitionPrior is None:
            TransitionPrior, TargetClassifier = _load_upstream_modules()
        self.k = k
        self.latents_per_block = latents_per_block
        self.c_hid = c_hid
        self.beta = beta
        self.beta_classifier = beta_classifier
        self.lambda_reg = lambda_reg
        self.gumbel_max = gumbel_max
        self.gumbel_min = gumbel_min
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)
        self.model: Optional[_CITRISVAE] = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, targets: np.ndarray) -> "CITRIS":
        """Train CITRIS-VAE on ``X`` ``(N, T, k)`` with intervention targets
        ``(N, T, k)`` (one-hot / all-zero per step; use ``mode="single"``).

        Builds ``(x_{t-1}, x_t, I_t)`` transition triplets and maximises the
        CITRIS-VAE ELBO: per-frame reconstruction + the upstream
        ``TransitionPrior.kl_divergence`` between the encoder posterior on
        ``x_t`` and the intervention-conditioned prior ``p(z_t | z_{t-1}, I_t)``.
        """
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        k = X.shape[2]
        if k != self.k:
            raise ValueError(f"X has {k} channels, expected k={self.k}")

        x_prev = X[:, :-1, :].reshape(-1, k)
        x_curr = X[:, 1:, :].reshape(-1, k)
        i_curr = targets[:, 1:, :].reshape(-1, k)  # I^t target (B, num_blocks)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_prev),
                torch.from_numpy(x_curr),
                torch.from_numpy(i_curr),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model = _CITRISVAE(
            self.k, self.latents_per_block, self.c_hid, self.lambda_reg
        ).to(self.device)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.history_ = []
        for ep in range(self.max_epochs):
            # Gumbel-Softmax temperature annealing (high -> low) so the
            # latent-to-causal assignment psi hardens over training, as in the
            # upstream CITRIS training schedule.
            frac = ep / max(self.max_epochs - 1, 1)
            tau = self.gumbel_max + (self.gumbel_min - self.gumbel_max) * frac
            self.model.prior.gumbel_temperature = tau
            self.model.intv_classifier.gumbel_temperature = tau
            epoch, nb = 0.0, 0
            for xb_prev, xb_curr, ib in loader:
                xb_prev = xb_prev.to(self.device)
                xb_curr = xb_curr.to(self.device)
                ib = ib.to(self.device)

                mu_t, ls_t = self.model.encode(xb_prev)
                z_t = self.model._reparam(mu_t, ls_t)
                mu_t1, ls_t1 = self.model.encode(xb_curr)
                z_t1 = self.model._reparam(mu_t1, ls_t1)

                # Per-frame reconstruction.
                rec = (
                    ((self.model.decode(z_t1) - xb_curr) ** 2).sum(dim=1)
                    + ((self.model.decode(z_t) - xb_prev) ** 2).sum(dim=1)
                )
                # Genuine CITRIS transition-prior KL (marginalised over psi).
                kld = self.model.prior.kl_divergence(
                    z_t=z_t, target=ib,
                    z_t1_mean=mu_t1, z_t1_logstd=ls_t1, z_t1_sample=z_t1,
                )
                loss = (rec + self.beta * kld).mean()

                # Genuine CITRIS target-classifier loss (specialises psi).
                # z_sample: (B, 2, num_latents); target: (B, 1, num_blocks).
                z_stack = torch.stack([z_t, z_t1], dim=1)
                loss_model, loss_z = self.model.intv_classifier(
                    z_sample=z_stack, target=ib[:, None, :],
                    transition_prior=self.model.prior, logger=None,
                )
                loss = loss + self.beta_classifier * (loss_model + loss_z)

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch += float(loss.item())
                nb += 1
            self.history_.append(epoch / max(nb, 1))
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Posterior-mean latents ``(N, T, num_latents)`` for ``X``."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        N, T, k = X.shape
        with torch.no_grad():
            flat = torch.from_numpy(X.reshape(-1, k)).to(self.device)
            mu, _ = self.model.encode(flat)
        return mu.cpu().numpy().reshape(N, T, self.model.num_latents)

    # ------------------------------------------------------------------
    def inferred_graph(
        self,
        max_lag: int = 1,
        threshold: float = 0.1,
        n_probe: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer the lag-1 causal adjacency from the trained transition prior.

        CITRIS-VAE does not output a causal graph directly. We derive one from
        the prior's input-sensitivity: for random ``z_{t-1}`` probes, the mean
        absolute Jacobian of each output latent's predicted prior-mean w.r.t.
        each input latent gives a latent×latent dependence, which we aggregate
        to causal blocks (channels) using the learned Gumbel-Softmax assignment
        ``psi`` (``block_sens = psiᵀ · J · psi``). Because block ``i`` is pinned
        to channel ``i`` by the intervention supervision, block indices map to
        channels.

        Returns
        -------
        adj_pred : (k, k, max_lag) int — 1 where the max-normalised block score
                   exceeds ``threshold`` at lag 1 (lags > 1 all-zero: the
                   first-order latent Markov prior).
        scores   : (k, k, max_lag) float in [0, 1]; ``[i, j, 0]`` = strength of
                   ``j -> i`` (benchmark convention ``adj[i, j, l]``: j causes i
                   at lag l+1). Self-loops zeroed (ground-truth excludes them).
        """
        self._check_fitted()
        m = self.model
        torch.manual_seed(self.seed + 1)
        z_prev = torch.randn(n_probe, m.num_latents, device=self.device, requires_grad=True)
        mu_p, _ = m.prior._get_prior_params(z_prev)  # (n_probe, num_latents)

        # Latent x latent sensitivity: |d mu_p[:, a] / d z_prev[:, b]|.
        J = np.zeros((m.num_latents, m.num_latents))
        for a in range(m.num_latents):
            g = torch.autograd.grad(mu_p[:, a].sum(), z_prev, retain_graph=True)[0]
            J[a] = g.abs().mean(dim=0).detach().cpu().numpy()

        # Aggregate latents -> causal blocks via the learned psi assignment
        # (drop the psi(0) noise column). psi: (num_latents, num_blocks+1).
        with torch.no_grad():
            psi = m.prior.get_target_assignment(hard=False).cpu().numpy()
        psi_causal = psi[:, : self.k]  # (num_latents, k)
        block = psi_causal.T @ J @ psi_causal  # (k, k): [i, j] = j -> i

        np.fill_diagonal(block, 0.0)
        smax = block.max()
        norm = block / smax if smax > 0 else block

        scores = np.zeros((self.k, self.k, max_lag))
        adj = np.zeros((self.k, self.k, max_lag), dtype=int)
        scores[:, :, 0] = norm
        adj[:, :, 0] = (norm > threshold).astype(int)
        return adj, scores

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("CITRIS is not fitted; call .fit(X, targets) first.")
