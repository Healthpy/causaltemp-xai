"""SCM-rollout recourse controls implemented by this project.

``NoiselessSCMRecourse`` and ``PearlSCMRecourse`` are positive controls whose
respective rollout-faithfulness properties hold by construction. They are not
implementations of a method from the external CARLA benchmarking library.

Before 2026-08-21 these classes and result-schema labels used ``CARLARecourse``
and ``PearlCARLARecourse`` / ``CARLA`` and ``PearlCARLA``. Those names were
removed because they incorrectly implied an implementation of CARLA
(Pawelczyk et al., 2021). Historical result artifacts retain the old labels so
published run evidence remains byte-stable; all new APIs and result schemas
use the SCM-specific names.

Two recourse variants are provided, differing only in the forward-rollout
semantics used to propagate the intervention past ``t0`` (see each class's
docstring):

* :class:`NoiselessSCMRecourse` — **noiseless**
  rollout: ``x_cf[t] = mechanism.forward_torch(window)`` for ``t > t0``. Scores
  ``cf_faith_rollout_hard == 1`` by construction
  (``CFfaith(semantics="noiseless_rollout")``), but the deterministic,
  noise-free continuation drifts off the noisy data manifold as the
  post-intervention horizon grows — the v0.1-documented long-horizon validity
  collapse (``docs/archive/hypotheses_assessment.md``).
* :class:`PearlSCMRecourse` (M2, added 2026-07-08) — **Pearl** rollout:
  abducts the exogenous noise from the factual trajectory
  (``eps[t] = x_orig[t] - mechanism.forward_numpy(window)``, exact under
  additive noise) and reuses it when rolling the recourse forward,
  ``x_cf[t] = mechanism.forward_torch(window) + eps[t]``. Scores
  ``cf_faith_pearl_hard == 1`` by construction
  (``CFfaith(semantics="pearl_delta")``) and stays on-manifold: it differs
  from the factual trajectory only by the propagated intervention rather than
  by a resampled/zeroed noise term.

  **Empirical finding (smoke-scale, 2026-07-08, honestly reported — not the
  naive expectation):** reinjecting noise does *not* unconditionally fix the
  noiseless control's long-horizon validity collapse. At its default
  ``lam_prox=0.5`` the Pearl variant performed worse (0.07 vs 0.40 at the
  longest smoke-scale horizon tested). The reason is structural, not a bug:
  the Pearl delta obeys the *homogeneous* recursion
  ``delta[t] = sum_l A_l @ delta[t-l]`` (the factual noise exactly cancels —
  see ``CFfaith``'s ``pearl_delta`` docstring), so under this benchmark's
  stability requirement (spectral radius < 1, ``generator._stabilise``) any
  one-shot intervention's effect decays geometrically and a proportionally
  *larger* ``delta[t0]`` is needed to survive to a long post-intervention
  horizon than the noiseless variant needs (whose raw *value*, not
  *delta*, follows the same contraction toward a class-independent fixed
  point that some instances land in "for free"). At the noiseless default
  ``lam_prox=0.5`` this larger ``delta[t0]`` is quadratically over-penalized.
  Lowering the default to ``lam_prox=0.1`` (this class's default; see
  ``__init__``) closes most of the gap: validity matches the noiseless variant at 3 of 4
  smoke-scale horizons tested and narrows the remaining gap at the most
  extreme one, while ``pearl_hard = 1.00`` holds at every horizon (by
  construction, unaffected by ``lam_prox``). See the M2 validation report for
  the full horizon-sweep numbers; this remains a smoke-scale finding (n=15
  instances, 1 seed) — full-scale confirmation is deferred to the PI's
  separate `full`/`full_nl` run. This paragraph records the historical fixed-
  penalty experiment. The current search still tries those defaults first but,
  after a failed search, makes one prediction-only attempt with early stopping
  and reports whether a genuine above-tolerance intervention was found.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from causaltemp_xai.benchmarks.mechanisms import lag_window as _lag_window
from causaltemp_xai.scm.intervention import INTERVENTION_TOL

# The first attempt always uses the configured ``lam_prox``. A failed search
# gets one prediction-only fallback with early stopping at the first genuine
# flip. This keeps worst-case optimiser work at 2x. Six successive halvings
# were tested on the current ``smoke_nl`` substrate and still produced 0/3
# flips, so a shallow geometric backoff only repeated the same local optimum.
PREDICTION_ONLY_FALLBACK_LAM = 0.0


def _resolve_t0_candidates(
    T: int,
    t0_fractions: tuple[float, ...],
    t0_steps: tuple[int, ...] | None = None,
) -> list[int]:
    """Resolve the intervention-timestep candidate set for a ``(T, k)`` instance.

    ``t0_steps`` is an **absolute** step index and takes precedence over
    ``t0_fractions`` when given. Both SCM recourse variants share this so a horizon
    sweep cannot silently pin one variant and not the other.

    Absolute is the correct unit for a horizon sweep (M4d): the Pearl delta
    decays geometrically in the *absolute* remaining horizon ``T - t0``, so a
    fraction of ``T`` is scale-dependent in the wrong direction — ``0.25``
    leaves 22 steps of decay at ``T = 30`` but 75 at ``T = 100``, which is why
    values validated at smoke scale are structurally hopeless at full scale
    (``docs/general_plan.md`` §10). Fractions are kept as the default so every
    pre-M4d run stays byte-identical.

    Candidates are clamped to ``0 < t0 < T - 1``. The lower bound is required by
    abduction (``eps[0]`` is never defined); the upper bound is the CF-faith
    degeneracy gate — a CF whose first change is at ``T - 1`` has no
    post-intervention trajectory to score and is NaN'd, so proposing it would
    only manufacture unscorable CFs. An empty result falls back to ``[T // 2]``.
    """
    if t0_steps is not None:
        proposed = sorted({int(t) for t in t0_steps})
        resolved = [t0 for t0 in proposed if 0 < t0 < T - 1]
        if proposed and not resolved:
            raise ValueError(
                f"no t0 in t0_steps={tuple(proposed)} is valid for T={T}: "
                f"require 0 < t0 < {T - 1}. Silently falling back would report a "
                "horizon the sweep did not actually evaluate."
            )
        return resolved or [T // 2]
    candidates = sorted({max(1, round(f * T)) for f in t0_fractions})
    return [t0 for t0 in candidates if 0 < t0 < T - 1] or [T // 2]


class NoiselessSCMRecourse:
    """Causal noiseless-rollout recourse generator (this project's own
    construction -- see the module docstring's naming disclosure).

    Optimises only ``x[t0]`` on the ``actionable_mask``-selected variables:

        delta_actionable = delta * actionable_mask
        x[t0] = x_orig[t0] + delta_actionable

    minimising ``lam_pred * CE(model(x_cf), target) + lam_prox * ||delta||^2``.
    The configured penalty is tried first; a failed search gets one prediction-
    only fallback with early stopping. It then deterministically rolls the mechanism forward
    for every ``t > t0``
    (:meth:`_rollout`) -- so the intervention propagates through the SCM by
    construction, not via a separately-encoded causal-ordering step. See
    :meth:`generate`'s docstring for the full contract.

    Parameters
    ----------
    target_class : int
        Desired output class.
    actionable_mask : ndarray or None
        Boolean mask of shape ``(T, k)`` or ``(k,)``.  ``True`` marks
        features that may be changed.  ``None`` means all features are
        actionable.
    lam_pred : float
        Prediction-loss weight.
    lam_prox : float
        Initial proximity regularisation weight. Successful first-round cases
        use this value exactly; failed searches use a bounded prediction-only fallback.
    lr : float
        Adam learning rate.
    n_steps : int
        Gradient-descent iterations.
    t0_fractions : tuple[float, ...]
        Candidate intervention timesteps as fractions of ``T``. The default
        ``(0.25, 0.5)`` is the locked benchmark setting; the fix adapts the
        penalty rather than selecting a different intervention horizon.
    t0_steps : tuple[int, ...] | None
        **Absolute** candidate intervention timesteps. When given, takes
        precedence over ``t0_fractions``. This is the M4d horizon-sweep entry
        point — see :func:`_resolve_t0_candidates` for why absolute rather than
        fractional is the correct unit. ``None`` (default) preserves
        pre-M4d behaviour exactly.
    """

    def __init__(
        self,
        target_class: int = 1,
        actionable_mask: Optional[np.ndarray] = None,
        lam_pred: float = 1.0,
        lam_prox: float = 0.5,
        lr: float = 0.05,
        n_steps: int = 500,
        t0_fractions: tuple[float, ...] = (0.25, 0.5),
        t0_steps: tuple[int, ...] | None = None,
    ) -> None:
        self.target_class = target_class
        self.actionable_mask = actionable_mask
        self.lam_pred = lam_pred
        self.lam_prox = lam_prox
        self.lr = lr
        self.n_steps = n_steps
        self.t0_fractions = t0_fractions
        self.t0_steps = t0_steps

    # ------------------------------------------------------------------
    # Differentiable noiseless VAR rollout
    # ------------------------------------------------------------------

    @staticmethod
    def _rollout(x_t: torch.Tensor, t0: int, x_t0: torch.Tensor, mechanism) -> torch.Tensor:
        """Build the CF trajectory: original up to ``t0``, free values at ``t0``,
        noiseless mechanism rollout after ``t0`` (differentiable w.r.t. ``x_t0``).

        ``x_cf[t] = mechanism.forward_torch(window)`` for ``t > t0`` (matching
        ``cf_faith.py``). Never re-injects noise â†’ the CF *is* its own noiseless
        rollout, so CFfaith(noiseless_rollout).hard == 1 by construction.
        """
        T = x_t.shape[0]
        L = mechanism.L
        rows: list[torch.Tensor] = []
        for t in range(T):
            if t < t0:
                rows.append(x_t[t])  # fixed original (zero retroactive change)
            elif t == t0:
                rows.append(x_t0)  # free / intervened values
            else:
                # Feed exactly L rows (oldestâ†’newest), zero-padding rows that
                # reach before t=0 (replicates the original ``if lag >= 0`` guard).
                window_rows = []
                for j in range(L):
                    src = t - L + j
                    window_rows.append(rows[src] if src >= 0 else torch.zeros_like(x_t0))
                window = torch.stack(window_rows, dim=0)  # (L, k)
                rows.append(mechanism.forward_torch(window))
        return torch.stack(rows, dim=0)  # (T, k)

    def _generate_one(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> tuple[np.ndarray, bool]:
        """Return ``(cf, found)`` for one instance.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)``
            Original time-series instance.
        model : LSTMClassifier
            Differentiable classifier exposing ``torch_logits``.
        graph : ndarray ``(k, k, L)``
            SCM adjacency (carried for API symmetry; not used directly here).
        mechanism : Mechanism
            Transition mechanism producing the deterministic next-step mean.

        A result is found only when it reaches ``target_class`` and its direct
        actionable intervention exceeds :data:`INTERVENTION_TOL`. Failed
        candidates are ranked by prediction loss, never by smallest edit.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        T, k = x_arr.shape
        x_t = torch.as_tensor(x_arr)
        target = torch.tensor([self.target_class], dtype=torch.long, device=model.device)

        if self.actionable_mask is None:
            mask = torch.ones(k, dtype=torch.float32)
        else:
            m = np.asarray(self.actionable_mask, dtype=np.float32)
            mask = torch.as_tensor(m[-1] if m.ndim == 2 else m)

        candidates = _resolve_t0_candidates(T, self.t0_fractions, self.t0_steps)

        attempted_lambdas: list[float] = []
        all_candidates: list[dict] = []
        best = None

        # Attempt the configured objective first. If it fails, use one
        # prediction-only fallback and stop at the first genuine flip. A
        # six-halving probe still found 0/3 flips on the current smoke_nl
        # substrate while costing 7x, so repeating shallow backoffs is neither
        # effective nor an acceptable production runtime.
        for backoff_round, effective_lam in enumerate(
            (self.lam_prox, PREDICTION_ONLY_FALLBACK_LAM)
        ):
            attempted_lambdas.append(float(effective_lam))
            round_candidates = []

            for t0 in candidates:
                delta = torch.zeros(k, requires_grad=True)
                optimiser = torch.optim.Adam([delta], lr=self.lr)
                x_orig_t0 = x_t[t0]

                for _step in range(self.n_steps):
                    optimiser.zero_grad()
                    actionable_delta = mask * delta
                    x_t0 = x_orig_t0 + actionable_delta
                    x_cf = self._rollout(x_t, t0, x_t0, mechanism)
                    logits = model.torch_logits(x_cf)
                    pred_loss = F.cross_entropy(logits, target)
                    prox = (actionable_delta**2).sum()
                    if (
                        backoff_round > 0
                        and int(logits.argmax(dim=1).item()) == self.target_class
                        and float(actionable_delta.detach().abs().max().item()) > INTERVENTION_TOL
                    ):
                        break
                    loss = self.lam_pred * pred_loss + effective_lam * prox
                    loss.backward()
                    optimiser.step()

                with torch.no_grad():
                    actionable_delta = mask * delta
                    x_t0 = x_orig_t0 + actionable_delta
                    x_cf = self._rollout(x_t, t0, x_t0, mechanism)
                    logits = model.torch_logits(x_cf)
                    pred_loss_val = float(F.cross_entropy(logits, target).item())
                    flipped = int(logits.argmax(dim=1).item()) == self.target_class
                    prox_val = float((actionable_delta**2).sum().item())
                    delta_max = float(actionable_delta.abs().max().item())
                    cf_arr = x_cf.detach().cpu().numpy().astype(np.float32)

                cand = {
                    "found": bool(flipped and delta_max > INTERVENTION_TOL),
                    "flipped": bool(flipped),
                    "pred_loss": pred_loss_val,
                    "proximity": prox_val,
                    "delta_max": delta_max,
                    "cf": cf_arr,
                    "t0": int(t0),
                    "backoff_round": int(backoff_round),
                    "effective_lam_prox": float(effective_lam),
                }
                round_candidates.append(cand)
                all_candidates.append(cand)

            successful = [cand for cand in round_candidates if cand["found"]]
            if successful:
                best = min(successful, key=lambda cand: cand["proximity"])
                break

        if best is None:
            # When nothing succeeds, retain the candidate closest to the target
            # boundary. The previous ``-proximity`` tie-break actively selected
            # the most degenerate no-op in this branch.
            best = min(all_candidates, key=lambda cand: (cand["pred_loss"], cand["proximity"]))

        self.last_generation_diagnostics = {
            "attempted_lam_prox": attempted_lambdas,
            "selected_round": best["backoff_round"],
            "selected_t0": best["t0"],
            "selected_effective_lam_prox": best["effective_lam_prox"],
            "selected_prediction_loss": best["pred_loss"],
            "selected_proximity": best["proximity"],
            "selected_delta_max": best["delta_max"],
            "classifier_target_reached": best["flipped"],
            "found": best["found"],
        }
        return best["cf"], bool(best["found"])

    def generate(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> np.ndarray:
        """Generate one CF while preserving the historical ndarray-only API."""
        cf, _found = self._generate_one(x, model, graph, mechanism)
        return cf

    # ------------------------------------------------------------------
    # CFExplainer alias interface + causal-info setter
    # ------------------------------------------------------------------

    def set_causal_info(self, graph, mechanism) -> None:
        """Store SCM graph and mechanism for use in fit/explain."""
        self._graph = graph
        self._mechanism = mechanism

    def fit(self, X_train, classifier) -> None:
        """No-op — NoiselessSCMRecourse needs graph/mechanism, set via set_causal_info()."""
        pass

    def explain(self, x, target_class: int, classifier) -> np.ndarray:
        """Alias for generate(x, classifier, self._graph, self._mechanism)."""
        graph = getattr(self, "_graph", None)
        mechanism = getattr(self, "_mechanism", None)
        if graph is None or mechanism is None:
            raise RuntimeError("Call set_causal_info(graph, mechanism) before explain().")
        return self.generate(x, classifier, graph, mechanism)

    def generate_batch(self, X: np.ndarray, model, graph=None, mechanism=None) -> np.ndarray:
        """Generate one CF per instance, preserving the ndarray-only API."""
        cfs, _no_cf_found = self.generate_batch_with_status(X, model, graph, mechanism)
        return cfs

    def generate_batch_with_status(
        self, X: np.ndarray, model, graph=None, mechanism=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return counterfactuals plus a Boolean ``no_cf_found`` vector."""
        X = np.asarray(X, dtype=np.float32)
        if graph is None:
            graph = getattr(self, "_graph", None)
        if mechanism is None:
            mechanism = getattr(self, "_mechanism", None)
        cfs, found, diagnostics = [], [], []
        for x in X:
            cf, found_i = self._generate_one(x, model, graph, mechanism)
            cfs.append(cf)
            found.append(found_i)
            diagnostics.append(dict(self.last_generation_diagnostics))
        self.last_batch_diagnostics = diagnostics
        return np.stack(cfs, axis=0), ~np.asarray(found, dtype=bool)


# =============================================================================
# Pearl-semantics variant (M2, 2026-07-08) — noise-reinjecting recourse
# =============================================================================


class PearlSCMRecourse:
    """Pearl-semantics causal recourse: noise-reinjecting counterfactual rollout.

    Same optimisation objective, actionability masking, and t0-candidate
    search as :class:`NoiselessSCMRecourse`, but the forward rollout reinjects the
    **abducted exogenous noise** from the factual trajectory
    (``eps[t] = x_orig[t] - mechanism.forward_numpy(window)``, exact under
    additive noise — identical construction to
    ``CFfaith(semantics="pearl_delta")``, see ``metrics/cf_faith.py``) instead
    of assuming a noiseless continuation. This makes the recourse an
    abduction-action-prediction counterfactual (Pearl's three steps: abduct
    the noise from the observed factual, act by fixing the intervened value,
    predict by rolling the mechanism forward while reusing the abducted
    noise) rather than a pure deterministic SCM continuation, so it scores
    ``CFfaith(semantics="pearl_delta").hard == 1`` by construction (instead of
    the base class's ``noiseless_rollout`` semantics) — targeting the
    v0.1-documented long-horizon validity collapse of the noiseless variant.

    **``lam_prox`` starts lower than ``NoiselessSCMRecourse`` (0.1 vs 0.5).** The
    Pearl delta obeys the homogeneous recursion
    ``delta[t] = sum_l A_l @ delta[t-l]`` (factual noise cancels exactly), so
    under this benchmark's stability requirement (spectral radius < 1) any
    one-shot intervention decays geometrically and needs a proportionally
    larger ``delta[t0]`` to still matter at a long horizon. NoiselessSCMRecourse's default
    ``lam_prox=0.5`` quadratically over-penalized that larger delta and,
    empirically (smoke-scale, 2026-07-08), turns a partial long-horizon
    validity problem into a near-total one (0.07 vs NoiselessSCMRecourse's 0.40 at the
    longest tested horizon). The current implementation preserves 0.1 as the
    first attempt, then applies the same bounded prediction-only fallback as
    the noiseless variant when no genuine flip is found. ``pearl_hard=1`` is unaffected (it is a property
    of the rollout construction, not of the optimisation weight).

    Deliberately **not** a subclass of / refactor into :class:`NoiselessSCMRecourse`:
    kept fully independent (duplicating the small optimisation-loop structure)
    so ``NoiselessSCMRecourse``'s pinned behaviour — ``tests/test_methods.py::
    TestNoiselessSCMRecourse`` and the M1-regenerated ``results/`` ``rollout_hard=1`` rows —
    carries zero refactor risk. This mirrors the codebase's existing
    convention of small per-class duplication over shared-base abstraction
    for CF generators (see e.g. ``methods/counterfactual/cfts_methods.py``,
    where every ``Cfts*CF`` class repeats its own ``generate_batch``).

    Parameters mirror :class:`NoiselessSCMRecourse` exactly (see its docstring)
    except the ``lam_prox`` default (0.1, not 0.5 — see above); the only
    behavioural difference otherwise is the forward-rollout semantics.
    """

    def __init__(
        self,
        target_class: int = 1,
        actionable_mask: Optional[np.ndarray] = None,
        lam_pred: float = 1.0,
        lam_prox: float = 0.1,
        lr: float = 0.05,
        n_steps: int = 500,
        t0_fractions: tuple[float, ...] = (0.25, 0.5),
        t0_steps: tuple[int, ...] | None = None,
    ) -> None:
        self.target_class = target_class
        self.actionable_mask = actionable_mask
        self.lam_pred = lam_pred
        self.lam_prox = lam_prox
        self.lr = lr
        self.n_steps = n_steps
        self.t0_fractions = t0_fractions
        self.t0_steps = t0_steps

    # ------------------------------------------------------------------
    # Noise abduction + Pearl rollout
    # ------------------------------------------------------------------

    @staticmethod
    def _abduct_noise(x_orig: np.ndarray, mechanism) -> np.ndarray:
        """Per-timestep exogenous noise abducted from the **factual**
        trajectory: ``eps[t] = x_orig[t] - mechanism.forward_numpy(window)``,
        for every ``t >= 1`` (``eps[0]`` stays 0 and is never used — the
        earliest usable ``t0`` is ``max(1, round(f*T)) >= 1``, and the
        rollout only ever reads ``eps[t]`` for ``t > t0 >= 1``). Exact under
        additive noise (see ``benchmarks.mechanisms.MLPMechanism`` module
        docstring's additivity contract); identical construction to
        ``CFfaith``'s ``pearl_delta`` abduction step
        (``metrics/cf_faith.py``), so this is not a new derivation — it is the
        same abduction reused for recourse instead of for scoring.
        """
        x_orig = np.asarray(x_orig, dtype=np.float32)
        T, k = x_orig.shape
        L = mechanism.L
        eps = np.zeros((T, k), dtype=np.float32)
        for t in range(1, T):
            window = _lag_window(x_orig, t, L, k)
            eps[t] = x_orig[t] - mechanism.forward_numpy(window)
        return eps

    @staticmethod
    def _rollout_pearl(
        x_t: torch.Tensor, t0: int, x_t0: torch.Tensor, mechanism, eps: torch.Tensor
    ) -> torch.Tensor:
        """Build the CF trajectory: original up to ``t0``, free values at
        ``t0``, Pearl (noise-reinjecting) mechanism rollout after ``t0``
        (differentiable w.r.t. ``x_t0``).

        ``x_cf[t] = mechanism.forward_torch(window) + eps[t]`` for ``t > t0``
        — same structure as ``NoiselessSCMRecourse._rollout`` but adds back the
        precomputed, non-differentiable ``eps[t]`` (constant w.r.t. the
        optimised ``delta``) at each forward step. Matches ``cf_faith.py``'s
        ``pearl_delta`` reference-trajectory construction exactly, so
        ``CFfaith(semantics="pearl_delta").hard == 1`` by construction.
        """
        T = x_t.shape[0]
        L = mechanism.L
        rows: list[torch.Tensor] = []
        for t in range(T):
            if t < t0:
                rows.append(x_t[t])  # fixed original (zero retroactive change)
            elif t == t0:
                rows.append(x_t0)  # free / intervened values
            else:
                window_rows = []
                for j in range(L):
                    src = t - L + j
                    window_rows.append(rows[src] if src >= 0 else torch.zeros_like(x_t0))
                window = torch.stack(window_rows, dim=0)  # (L, k)
                rows.append(mechanism.forward_torch(window) + eps[t])
        return torch.stack(rows, dim=0)  # (T, k)

    def _generate_one(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> tuple[np.ndarray, bool]:
        """Return ``(cf, found)`` for one Pearl-faithful search.

        Same signature, actionability masking, t0-candidate search, and
        Adam-optimised objective as ``NoiselessSCMRecourse.generate`` — only the
        rollout used inside the optimisation loop differs (Pearl noise
        reinjection instead of a noiseless continuation).
        """
        x_arr = np.asarray(x, dtype=np.float32)
        T, k = x_arr.shape
        x_t = torch.as_tensor(x_arr)
        target = torch.tensor([self.target_class], dtype=torch.long, device=model.device)

        eps_np = self._abduct_noise(x_arr, mechanism)
        eps = torch.as_tensor(eps_np)

        if self.actionable_mask is None:
            mask = torch.ones(k, dtype=torch.float32)
        else:
            m = np.asarray(self.actionable_mask, dtype=np.float32)
            mask = torch.as_tensor(m[-1] if m.ndim == 2 else m)

        candidates = _resolve_t0_candidates(T, self.t0_fractions, self.t0_steps)

        attempted_lambdas: list[float] = []
        all_candidates: list[dict] = []
        best = None

        for backoff_round, effective_lam in enumerate(
            (self.lam_prox, PREDICTION_ONLY_FALLBACK_LAM)
        ):
            attempted_lambdas.append(float(effective_lam))
            round_candidates = []

            for t0 in candidates:
                delta = torch.zeros(k, requires_grad=True)
                optimiser = torch.optim.Adam([delta], lr=self.lr)
                x_orig_t0 = x_t[t0]

                for _step in range(self.n_steps):
                    optimiser.zero_grad()
                    actionable_delta = mask * delta
                    x_t0 = x_orig_t0 + actionable_delta
                    x_cf = self._rollout_pearl(x_t, t0, x_t0, mechanism, eps)
                    logits = model.torch_logits(x_cf)
                    pred_loss = F.cross_entropy(logits, target)
                    prox = (actionable_delta**2).sum()
                    if (
                        backoff_round > 0
                        and int(logits.argmax(dim=1).item()) == self.target_class
                        and float(actionable_delta.detach().abs().max().item()) > INTERVENTION_TOL
                    ):
                        break
                    loss = self.lam_pred * pred_loss + effective_lam * prox
                    loss.backward()
                    optimiser.step()

                with torch.no_grad():
                    actionable_delta = mask * delta
                    x_t0 = x_orig_t0 + actionable_delta
                    x_cf = self._rollout_pearl(x_t, t0, x_t0, mechanism, eps)
                    logits = model.torch_logits(x_cf)
                    pred_loss_val = float(F.cross_entropy(logits, target).item())
                    flipped = int(logits.argmax(dim=1).item()) == self.target_class
                    prox_val = float((actionable_delta**2).sum().item())
                    delta_max = float(actionable_delta.abs().max().item())
                    cf_arr = x_cf.detach().cpu().numpy().astype(np.float32)

                cand = {
                    "found": bool(flipped and delta_max > INTERVENTION_TOL),
                    "flipped": bool(flipped),
                    "pred_loss": pred_loss_val,
                    "proximity": prox_val,
                    "delta_max": delta_max,
                    "cf": cf_arr,
                    "t0": int(t0),
                    "backoff_round": int(backoff_round),
                    "effective_lam_prox": float(effective_lam),
                }
                round_candidates.append(cand)
                all_candidates.append(cand)

            successful = [cand for cand in round_candidates if cand["found"]]
            if successful:
                best = min(successful, key=lambda cand: cand["proximity"])
                break

        if best is None:
            best = min(all_candidates, key=lambda cand: (cand["pred_loss"], cand["proximity"]))

        self.last_generation_diagnostics = {
            "attempted_lam_prox": attempted_lambdas,
            "selected_round": best["backoff_round"],
            "selected_t0": best["t0"],
            "selected_effective_lam_prox": best["effective_lam_prox"],
            "selected_prediction_loss": best["pred_loss"],
            "selected_proximity": best["proximity"],
            "selected_delta_max": best["delta_max"],
            "classifier_target_reached": best["flipped"],
            "found": best["found"],
        }
        return best["cf"], bool(best["found"])

    def generate(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> np.ndarray:
        """Generate one CF while preserving the historical ndarray-only API."""
        cf, _found = self._generate_one(x, model, graph, mechanism)
        return cf

    def generate_batch(
        self, X: np.ndarray, model, graph: np.ndarray = None, mechanism=None
    ) -> np.ndarray:
        """Generate one CF per instance, preserving the ndarray-only API."""
        cfs, _no_cf_found = self.generate_batch_with_status(X, model, graph, mechanism)
        return cfs

    def generate_batch_with_status(
        self, X: np.ndarray, model, graph: np.ndarray = None, mechanism=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return counterfactuals plus a Boolean ``no_cf_found`` vector."""
        X = np.asarray(X, dtype=np.float32)
        if graph is None:
            graph = getattr(self, "_graph", None)
        if mechanism is None:
            mechanism = getattr(self, "_mechanism", None)
        cfs, found, diagnostics = [], [], []
        for x in X:
            cf, found_i = self._generate_one(x, model, graph, mechanism)
            cfs.append(cf)
            found.append(found_i)
            diagnostics.append(dict(self.last_generation_diagnostics))
        self.last_batch_diagnostics = diagnostics
        return np.stack(cfs, axis=0), ~np.asarray(found, dtype=bool)

    # ------------------------------------------------------------------
    # CFExplainer alias interface + causal-info setter (parity with NoiselessSCMRecourse)
    # ------------------------------------------------------------------

    def set_causal_info(self, graph, mechanism) -> None:
        """Store SCM graph and mechanism for use in fit/explain."""
        self._graph = graph
        self._mechanism = mechanism

    def fit(self, X_train, classifier) -> None:
        """No-op — PearlSCMRecourse needs graph/mechanism, set via set_causal_info()."""
        pass

    def explain(self, x, target_class: int, classifier) -> np.ndarray:
        """Alias for generate(x, classifier, self._graph, self._mechanism)."""
        graph = getattr(self, "_graph", None)
        mechanism = getattr(self, "_mechanism", None)
        if graph is None or mechanism is None:
            raise RuntimeError("Call set_causal_info(graph, mechanism) before explain().")
        return self.generate(x, classifier, graph, mechanism)
