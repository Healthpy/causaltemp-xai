"""SCM-regularised counterfactual explainer (Bahri, Li, Hosseinzadeh, Filali
Boubrahimi & Hamdi, *Improving Causal Feasibility in Counterfactual
Explanations for Multivariate Time Series Classification*, IEEE BigData 2025,
DOI 10.1109/BIGDATA66926.2025.11402392).

**Class renamed 2026-08-06** (PI decision) from
``CausalFeasibilityCF`` to ``TSCausalCF``, and wired into the main
experiment's default method list (previously opt-in only via
``--methods CausalFeasibility``). The rename is cosmetic -- the
implementation, faithfulness verification, and R3 status below are
unchanged; the registry key is now ``"TSCausal"`` (was ``"CausalFeasibility"``).

Read from the paper's own PDF (not a secondhand summary) before implementing
-- see 2026-08-04. That reading corrected an earlier
misreading in this project's own docs: there is **no intervention timestep**
in this method. ``docs/risk_register.md`` RISK-15 had it right ("no do()
operator, no intervention time, no abduction"); an earlier draft of the
M3 formula used the notation ``U_d@t0`` in a way that read like
this benchmark's own do()-abduction moment, which the paper does not have.
"t0" in the paper's own equations means the literal first observed timestep
``t=0``, nothing more.

The method (their eq. 5), reproduced exactly::

    argmin_delta  L_pred(f(T + delta), target)
                + L_prox(delta, U_s, U_d)          # eq. 3
                + L_causal(T, delta, V, U_d)        # eq. 4

* ``delta`` is a **free** ``(T, k)`` perturbation over the *entire* trajectory
  -- there is no "fixed before / free after" split the way this benchmark's
  own ``NoiselessSCMRecourse`` has. ``T' = T + delta`` is optimised directly.
* ``L_prox`` (eq. 3): an L1/Lp proximity penalty on ``U_s`` (every timestep)
  and on ``U_d`` **at t=0 only** -- the one moment those variables have no
  causal parents to be checked against (eq. 2: ``Pa(T_tau^(v)) = T_{t<tau}^(v)``
  for ``v in U_d``, which is vacuous at ``tau=0``).
* ``L_causal`` (eq. 4): a residual ``|T'_t^(v) - f(Pa(T'_t^(v)))|`` against the
  SCM's own structural equation, applied to every ``V`` channel at **every**
  timestep ``t`` (including ``t=0``, zero-padding the missing pre-window
  history the same way this benchmark's ``lag_window``/``NoiselessSCMRecourse``
  already do) and to ``U_d`` channels for ``t >= 1``.
* Optimised via FISTA (Beck & Teboulle, 2009) -- gradient descent on the
  smooth part (``L_pred + L_causal``) with a proximal soft-threshold step on
  the L1 ``L_prox`` term and Nesterov-style momentum, not plain Adam. The
  paper is explicit about the algorithm ("following FISTA... as described in
  [17]", their [17] being van Looveren & Klaise's CEGP), and this project's
  R3 rule means an Adam-with-L1-folded-into-the-loss stand-in would not have
  earned this class the paper's name.

U_s/U_d/V typing (2026-08-04): ``U_d`` =
:func:`causaltemp_xai.benchmarks.generator.exogenous_channels` (channels with
no incoming edge at any lag); ``U_s`` is always empty -- this benchmark's
``(T, k)`` tensor has no separate static-covariate slot, matching the paper's
own formalism where ``U_s`` variables are still part of the ``(V, L)`` array,
just held constant. On ``full``/``full_nl`` (paper-scale, k=10, sparsity=0.2)
``U_d`` is also empty -- every channel has a parent -- so the objective
degenerates to ``L_pred + lambda * L_causal(all channels, every t)`` with no
proximal step ever firing (nothing sits in ``U_s`` union ``U_d@t=0`` to
shrink). This is accepted, not routed around: it is what this benchmark's
paper-scale graph density honestly produces, and per M3 this
method is a third-party foil being critiqued (``docs/risk_register.md``
RISK-16), not one this benchmark owes a maximally favourable partition.
``smoke`` has one exogenous channel, so the proximal step is exercised there.

``p`` (the paper's norm order, always 1 in their experiments) is accepted for
interface parity but has no numerical effect here: this benchmark's channels
are scalar (unlike the paper's own vector-valued position/velocity channels,
where an L1 vs L2 norm over the 2-D vector genuinely differs), so ``||x||_p``
for a scalar ``x`` is ``|x|`` for every ``p``.

Their ``lambda_s=1``, ``lambda_d=lambda_v=lambda=13`` (their own
proximity/feasibility trade-off study, Table/Fig. 4) are this class's
defaults. This is a **third-party baseline to critique, not a positive
control** -- unlike ``NoiselessSCMRecourse``/``PearlSCMRecourse``, its faithfulness
under this benchmark's own CF-faith metric is not true by construction, and
that is the point (M3, ``docs/risk_register.md`` RISK-16: its
own causal-likelihood criterion cannot see intervention-level failure, so
``Delta_outcome`` and CF-faith scored on it are the headline third-party
test).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from causaltemp_xai.benchmarks.generator import exogenous_channels


def _shift_forward(x: torch.Tensor, shift: int) -> torch.Tensor:
    """``x`` shifted forward by ``shift`` rows, zero-filling the front.

    ``result[t] = x[t - shift]`` when that index is ``>= 0``, else 0. The
    building block for a fully vectorised (no per-timestep Python loop)
    lag-window construction, matching
    :func:`causaltemp_xai.benchmarks.mechanisms.lag_window`'s zero-pad
    convention (a lag reaching before ``t=0`` contributes nothing).
    """
    T = x.shape[0]
    if shift <= 0:
        return x
    if shift >= T:
        return torch.zeros_like(x)
    pad = torch.zeros(shift, *x.shape[1:], dtype=x.dtype)
    return torch.cat([pad, x[: T - shift]], dim=0)


def _batched_lag_windows(x: torch.Tensor, L: int) -> torch.Tensor:
    """``(T, k) -> (T, L, k)``: window ``[t]`` is the L-lag history feeding a
    prediction of ``x[t]``, oldest to newest (``window[t, -1] = x[t - 1]``),
    zero-padded before ``t = 0``. Vectorised over every ``t`` in one shot --
    unlike :class:`~causaltemp_xai.methods.counterfactual.scm_recourse.NoiselessSCMRecourse`,
    this method has no sequential rollout, so every timestep's window can be
    built from the free ``x`` directly rather than one row at a time.
    """
    rows = [_shift_forward(x, L - j) for j in range(L)]
    return torch.stack(rows, dim=1)  # (T, L, k)


def _soft_threshold(z: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    """Proximal operator of ``thresh * |.|_1``: elementwise soft-thresholding."""
    return torch.sign(z) * torch.clamp(z.abs() - thresh, min=0.0)


def _build_loss_masks(
    T: int, k: int, u_s: list[int], u_d: list[int], lam_s: float, lam_d: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-cell (t, channel) loss assignment (eqs. 3-4): which cells are
    proximity-only (L1 shrunk, no causal-residual term) versus causally
    checked, and at what shrinkage threshold.

    ``prox_mask[t, j]`` is True exactly where the paper's ``L_prox`` applies:
    ``U_s`` at *every* timestep, ``U_d`` at ``t = 0`` only -- the one moment
    those variables have no causal parents to check against. Every other
    cell (``V`` at every ``t``; ``U_d`` at ``t >= 1``) gets ``L_causal``
    instead -- ``causal_mask`` is exactly the complement, not a separately
    reasoned-about set, so the two can never silently disagree about a cell.

    Extracted from :meth:`TSCausalCF.generate` as a pure function
    so the (t, channel) assignment can be tested directly, without depending
    on gradient/optimisation dynamics to observe it indirectly.
    """
    prox_mask = torch.zeros(T, k, dtype=torch.bool)
    if u_s:
        prox_mask[:, u_s] = True
    if u_d:
        prox_mask[0, u_d] = True
    causal_mask = ~prox_mask

    thresh = torch.zeros(T, k)
    if u_s:
        thresh[:, u_s] = lam_s
    if u_d:
        thresh[0, u_d] = lam_d

    return prox_mask, causal_mask, thresh


class TSCausalCF:
    """SCM-regularised counterfactual (Bahri et al., IEEE BigData 2025).

    Parameters
    ----------
    target_class:
        Desired output class.
    lam:
        ``lambda_d = lambda_v = lambda`` -- shared weight on the dynamic-
        exogenous proximity term and the causal-residual term (paper default
        13.0, their own tuned trade-off point on dataset D1).
    lam_s:
        ``lambda_s`` -- weight on the static-exogenous proximity term (paper
        default 1.0). Never actually multiplies anything in this benchmark,
        since ``U_s`` is always empty here -- kept for interface parity and
        so a future benchmark variant with real static-exogenous channels
        does not need a new parameter.
    p:
        Norm order for both loss terms (paper default 1; they only ever use
        ``p=1``). No numerical effect here -- see module docstring.
    lr:
        FISTA step size.
    n_steps:
        Fixed number of FISTA iterations (this codebase's existing
        convention for gradient-based CF methods -- see
        :class:`~causaltemp_xai.methods.counterfactual.scm_recourse.NoiselessSCMRecourse` --
        rather than the paper's Figure 2 "loop until flipped").
    """

    def __init__(
        self,
        target_class: int = 1,
        lam: float = 13.0,
        lam_s: float = 1.0,
        p: int = 1,
        lr: float = 0.05,
        n_steps: int = 500,
    ) -> None:
        self.target_class = target_class
        self.lam = lam
        self.lam_s = lam_s
        self.p = p
        self.lr = lr
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    # FISTA optimisation
    # ------------------------------------------------------------------

    def generate(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> np.ndarray:
        """Generate an SCM-regularised counterfactual for instance ``x``.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)``
        model : LSTMClassifier
            Differentiable classifier exposing ``torch_logits``.
        graph : ndarray ``(k, k, L)``
            SCM adjacency -- used to derive the ``U_d`` typing
            (:func:`exogenous_channels`).
        mechanism : Mechanism
            Transition mechanism producing the deterministic next-step mean.

        Returns
        -------
        cf : ndarray of shape ``(T, k)``, ``x + delta`` at the final FISTA
            iterate.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        T, k = x_arr.shape
        x_t = torch.as_tensor(x_arr)
        target = torch.tensor([self.target_class], dtype=torch.long, device=model.device)

        u_d = exogenous_channels(graph)
        u_s: list[int] = []  # always empty -- see module docstring

        prox_mask, causal_mask, thresh = _build_loss_masks(T, k, u_s, u_d, self.lam_s, self.lam)
        causal_mask_f = causal_mask.to(torch.float32)

        delta = torch.zeros(T, k, dtype=torch.float32)
        y = delta.clone().requires_grad_(True)

        for step in range(self.n_steps):
            x_cf = x_t + y
            logits = model.torch_logits(x_cf)
            pred_loss = F.cross_entropy(logits, target)

            windows = _batched_lag_windows(x_cf, mechanism.L)  # (T, L, k)
            f_pa = mechanism.forward_torch(windows)  # (T, k)
            resid = (x_cf - f_pa).abs()  # scalar channels: ||.||_p == |.| for any p
            causal_loss = (resid * causal_mask_f).sum()

            smooth_loss = pred_loss + self.lam * causal_loss
            (grad,) = torch.autograd.grad(smooth_loss, y)

            with torch.no_grad():
                z = y - self.lr * grad
                delta_new = torch.where(prox_mask, _soft_threshold(z, self.lr * thresh), z)
                momentum = step / (step + 3)  # FISTA/CEGP fixed-iteration schedule
                y_next = delta_new + momentum * (delta_new - delta)
                delta = delta_new

            y = y_next.detach().requires_grad_(True)

        with torch.no_grad():
            x_cf = x_t + delta
            cf_arr = x_cf.cpu().numpy().astype(np.float32)

        return cf_arr

    # ------------------------------------------------------------------
    # CFExplainer alias interface + causal-info setter (parity with NoiselessSCMRecourse)
    # ------------------------------------------------------------------

    def set_causal_info(self, graph, mechanism) -> None:
        """Store SCM graph and mechanism for use in fit/explain."""
        self._graph = graph
        self._mechanism = mechanism

    def fit(self, X_train, classifier) -> None:
        """No-op -- TSCausalCF needs graph/mechanism, set via set_causal_info()."""
        pass

    def explain(self, x, target_class: int, classifier) -> np.ndarray:
        """Alias for generate(x, classifier, self._graph, self._mechanism)."""
        graph = getattr(self, "_graph", None)
        mechanism = getattr(self, "_mechanism", None)
        if graph is None or mechanism is None:
            raise RuntimeError("Call set_causal_info(graph, mechanism) before explain().")
        return self.generate(x, classifier, graph, mechanism)

    def generate_batch(
        self, X: np.ndarray, model, graph: Optional[np.ndarray] = None, mechanism=None
    ) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        X = np.asarray(X, dtype=np.float32)
        if graph is None:
            graph = getattr(self, "_graph", None)
        if mechanism is None:
            mechanism = getattr(self, "_mechanism", None)
        return np.stack([self.generate(x, model, graph, mechanism) for x in X], axis=0)
