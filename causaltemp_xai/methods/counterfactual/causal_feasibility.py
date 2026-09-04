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
        Initial FISTA step size (``1 / L_0``). With ``backtrack=True`` this is
        an upper bound that the line search shrinks as needed, not a fixed
        step.
    n_steps:
        Maximum number of FISTA iterations (this codebase's existing
        convention for gradient-based CF methods -- see
        :class:`~causaltemp_xai.methods.counterfactual.scm_recourse.NoiselessSCMRecourse` --
        rather than the paper's Figure 2 "loop until flipped"). With
        ``select_best=True`` this is a budget rather than a result-defining
        knob: the returned iterate is the best one seen, so a larger budget
        can only help. Raised 500 -> 1000 alongside the line search: a
        correctly-stepped run makes smaller, admissible moves than the old
        diverging one, so it needs more of them (measured: first flip at
        step 44-203 on the ``smoke`` fixture, converged causal residual by
        ~1000).
    huber_beta:
        Half-width of the quadratic region of the Huberised causal residual
        used **for the gradient only** (0 disables it, restoring a bare
        ``|.|``). FISTA assumes the smooth part has a Lipschitz-continuous
        gradient; ``|r|`` does not, and with the paper's ``lambda=13`` a
        fixed-step proximal-gradient iteration on it does not converge -- it
        drifts with a per-cell step of ``lr * lambda`` that never decays. The
        reported/selected objective is always the paper's true L1 one.
    backtrack:
        Enable the Beck & Teboulle (2009, sec. 4) backtracking line search.
        The Lipschitz constant here depends on the classifier, the mechanism,
        ``lambda`` and the data scale, so no single fixed ``lr`` is safe
        across presets.
    eta:
        Step-shrink factor for the line search.
    grow:
        Per-iteration re-growth factor for the backtracked step, capped at
        ``lr``. Beck & Teboulle's own backtracking only ever *shrinks* the
        step, so one badly-conditioned iterate permanently throttles the rest
        of the run; letting it recover between iterations keeps the same
        per-step descent guarantee while restoring the step size once the
        iterate leaves the sharp region.
    min_lr:
        Floor on the backtracked step size (guards against an infinite
        shrink loop at a non-differentiable point).
    select_best:
        Return the best iterate seen -- any flipped one, and among flipped
        the lowest-objective one -- rather than the last. Set ``False`` to
        recover the previous last-iterate behaviour.
    """

    def __init__(
        self,
        target_class: int = 1,
        lam: float = 13.0,
        lam_s: float = 1.0,
        p: int = 1,
        lr: float = 0.05,
        n_steps: int = 1000,
        huber_beta: float = 0.1,
        backtrack: bool = True,
        eta: float = 2.0,
        grow: float = 1.2,
        min_lr: float = 1e-8,
        select_best: bool = True,
    ) -> None:
        self.target_class = target_class
        self.lam = lam
        self.lam_s = lam_s
        self.p = p
        self.lr = lr
        self.n_steps = n_steps
        self.huber_beta = huber_beta
        self.backtrack = backtrack
        self.eta = eta
        self.grow = grow
        self.min_lr = min_lr
        self.select_best = select_best

    # ------------------------------------------------------------------
    # FISTA optimisation
    # ------------------------------------------------------------------

    def _run_fista(self, x, model, graph, mechanism):
        """The FISTA loop. Returns ``(cf, flipped, info)``.

        ``flipped`` is whether the returned iterate is actually classified as
        ``target_class`` -- measured on the returned iterate, not inferred.
        ``info`` carries the objective decomposition and the backtracked step
        size, for diagnostics.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        T, k = x_arr.shape
        x_t = torch.as_tensor(x_arr)
        target = torch.tensor([self.target_class], dtype=torch.long, device=model.device)

        u_d = exogenous_channels(graph)
        u_s: list[int] = []  # always empty -- see module docstring

        prox_mask, causal_mask, thresh = _build_loss_masks(T, k, u_s, u_d, self.lam_s, self.lam)
        causal_mask_f = causal_mask.to(torch.float32)

        def smooth(d: torch.Tensor):
            """``L_pred + lambda * L_causal`` at ``delta = d``.

            Returns ``(surrogate, true_value, logits)``. ``surrogate`` is what
            FISTA differentiates: the causal residual is Huberised (scaled so
            it agrees with ``|.|`` outside ``|r| < beta``) because FISTA's
            convergence needs the smooth part to have a Lipschitz-continuous
            gradient, and a bare ``|.|`` residual does not -- its subgradient
            has constant magnitude right up to the kink, so a proximal-
            gradient step never settles and instead oscillates with amplitude
            ``lr * lambda``. ``true_value`` is the paper's own L1 objective,
            used for iterate selection and reporting so nothing downstream
            ever sees the surrogate's number.
            """
            x_cf = x_t + d
            logits = model.torch_logits(x_cf)
            pred_loss = F.cross_entropy(logits, target)
            windows = _batched_lag_windows(x_cf, mechanism.L)  # (T, L, k)
            f_pa = mechanism.forward_torch(windows)  # (T, k)
            resid = x_cf - f_pa
            # scalar channels: ||.||_p == |.| for any p (module docstring)
            causal_l1 = (resid.abs() * causal_mask_f).sum()
            if self.huber_beta > 0:
                huber = (
                    F.smooth_l1_loss(
                        resid,
                        torch.zeros_like(resid),
                        beta=self.huber_beta,
                        reduction="none",
                    )
                    / self.huber_beta
                )
                causal_s = (huber * causal_mask_f).sum()
            else:
                causal_s = causal_l1
            return (
                pred_loss + self.lam * causal_s,
                (pred_loss + self.lam * causal_l1).detach(),
                logits.detach(),
            )

        def prox_penalty(d: torch.Tensor) -> torch.Tensor:
            """``L_prox`` (eq. 3). ``thresh`` is zero off ``prox_mask``, so the
            elementwise product already restricts the sum to those cells."""
            return (d.abs() * thresh).sum()

        delta = torch.zeros(T, k, dtype=torch.float32)
        y = delta.clone().requires_grad_(True)
        t_mom = 1.0
        s = float(self.lr)

        best_obj = float("inf")
        best_delta = delta.clone()
        best_flipped = False
        first_flip_step = -1
        last: dict = {}

        for _step in range(self.n_steps):
            s = min(float(self.lr), s * float(self.grow))
            g_y, _, _ = smooth(y)
            (grad,) = torch.autograd.grad(g_y, y)
            g_y_val = float(g_y.detach())

            # Backtracking line search (Beck & Teboulle 2009, sec. 4): the
            # Lipschitz constant of the smooth part is not known here -- it
            # depends on the classifier, the mechanism, lambda and the data
            # scale -- and a fixed step above 1/L diverges instead of
            # converging, which is what the previous fixed ``lr`` did.
            while True:
                with torch.no_grad():
                    z = y - s * grad
                    cand = torch.where(prox_mask, _soft_threshold(z, s * thresh), z)
                    diff = cand - y
                    g_c, true_c, logits_c = smooth(cand)
                    q = g_y_val + float((grad * diff).sum()) + float(diff.pow(2).sum()) / (2 * s)
                if not self.backtrack or float(g_c) <= q + 1e-9 or s <= self.min_lr:
                    break
                s /= self.eta

            with torch.no_grad():
                delta_new = cand
                obj = float(true_c + prox_penalty(delta_new))
                flipped = bool(int(logits_c.argmax(dim=-1).item()) == self.target_class)
                # Prefer any flipped iterate; among flipped, the lowest
                # objective. Without this the method returns whatever the
                # fixed iteration budget happened to land on, which for a
                # diverging run can be an iterate that has already flipped
                # *back* -- making both validity and proximity a function of
                # ``n_steps`` rather than of the objective.
                better = (flipped and not best_flipped) or (
                    flipped == best_flipped and obj < best_obj
                )
                if flipped and first_flip_step < 0:
                    first_flip_step = _step
                if better:
                    best_obj, best_delta, best_flipped = obj, delta_new.clone(), flipped

                t_next = (1.0 + (1.0 + 4.0 * t_mom * t_mom) ** 0.5) / 2.0
                y_next = delta_new + ((t_mom - 1.0) / t_next) * (delta_new - delta)
                t_mom = t_next
                delta = delta_new
                last = {"objective": obj, "flipped": flipped, "step_size": s}

            y = y_next.detach().requires_grad_(True)

        with torch.no_grad():
            out = best_delta if self.select_best else delta
            out_flipped = best_flipped if self.select_best else bool(last.get("flipped", False))
            cf_arr = (x_t + out).cpu().numpy().astype(np.float32)

        info = dict(last)
        info["best_objective"] = best_obj
        info["first_flip_step"] = first_flip_step
        return cf_arr, out_flipped, info

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
        cf : ndarray of shape ``(T, k)``, ``x + delta`` at the best FISTA
            iterate (see ``select_best``).
        """
        cf, _flipped, _info = self._run_fista(x, model, graph, mechanism)
        return cf

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

    def generate_batch_with_status(
        self, X: np.ndarray, model, graph: Optional[np.ndarray] = None, mechanism=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``generate_batch`` plus a measured failed-search flag per instance.

        The flip check is already computed inside the FISTA loop (it drives
        best-iterate selection), so reporting it costs nothing -- and Phase 03
        no longer has to record this method's ``no_cf_found`` provenance as
        ``inferred``.
        """
        X = np.asarray(X, dtype=np.float32)
        if graph is None:
            graph = getattr(self, "_graph", None)
        if mechanism is None:
            mechanism = getattr(self, "_mechanism", None)
        cfs, found = [], []
        for x in X:
            cf, flipped, _info = self._run_fista(x, model, graph, mechanism)
            cfs.append(cf)
            found.append(flipped)
        return np.stack(cfs, axis=0), ~np.asarray(found, dtype=bool)
