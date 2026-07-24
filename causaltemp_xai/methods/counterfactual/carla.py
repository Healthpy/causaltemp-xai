"""CARLA-style causal recourse â€” stub.

Reference
---------
Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter, T., & Kasneci, G.
(2021). *CARLA: A Python Library to Benchmark Algorithmic Recourse and
Counterfactual Explanation Algorithms.*  NeurIPS 2021 Datasets and Benchmarks.

This stub defines the ``CARLARecourse`` class interface.  The real
implementation would train a generative model (e.g. CVAE) or apply
constrained optimisation that respects actionability constraints and the
causal ordering over variables.

Two recourse variants are provided, differing only in the forward-rollout
semantics used to propagate the intervention past ``t0`` (see each class's
docstring):

* :class:`CARLARecourse` (default, unchanged since v0.1) — **noiseless**
  rollout: ``x_cf[t] = mechanism.forward_torch(window)`` for ``t > t0``. Scores
  ``cf_faith_rollout_hard == 1`` by construction
  (``CFfaith(semantics="noiseless_rollout")``), but the deterministic,
  noise-free continuation drifts off the noisy data manifold as the
  post-intervention horizon grows — the v0.1-documented long-horizon validity
  collapse (``docs/hypotheses_assessment.md``).
* :class:`PearlCARLARecourse` (M2, added 2026-07-08) — **Pearl** rollout:
  abducts the exogenous noise from the factual trajectory
  (``eps[t] = x_orig[t] - mechanism.forward_numpy(window)``, exact under
  additive noise) and reuses it when rolling the recourse forward,
  ``x_cf[t] = mechanism.forward_torch(window) + eps[t]``. Scores
  ``cf_faith_pearl_hard == 1`` by construction
  (``CFfaith(semantics="pearl_delta")``) and stays on-manifold: it differs
  from the factual trajectory only by the propagated intervention rather than
  by a resampled/zeroed noise term.

  **Empirical finding (smoke-scale, 2026-07-08, honestly reported — not the
  naive expectation):** reinjecting noise does *not* unconditionally "fix"
  CARLARecourse's long-horizon validity collapse; at CARLA's own default
  ``lam_prox=0.5`` it makes the collapse *worse* (0.07 vs CARLA's 0.40 at the
  longest smoke-scale horizon tested). The reason is structural, not a bug:
  the Pearl delta obeys the *homogeneous* recursion
  ``delta[t] = sum_l A_l @ delta[t-l]`` (the factual noise exactly cancels —
  see ``CFfaith``'s ``pearl_delta`` docstring), so under this benchmark's
  stability requirement (spectral radius < 1, ``generator._stabilise``) any
  one-shot intervention's effect decays geometrically and a proportionally
  *larger* ``delta[t0]`` is needed to survive to a long post-intervention
  horizon than CARLA's noiseless variant needs (whose raw *value*, not
  *delta*, follows the same contraction toward a class-independent fixed
  point that some instances land in "for free"). At CARLA's default
  ``lam_prox=0.5`` this larger ``delta[t0]`` is quadratically over-penalized.
  Lowering the default to ``lam_prox=0.1`` (this class's default; see
  ``__init__``) closes most of the gap: validity matches CARLA at 3 of 4
  smoke-scale horizons tested and narrows the remaining gap at the most
  extreme one, while ``pearl_hard = 1.00`` holds at every horizon (by
  construction, unaffected by ``lam_prox``). See the M2 validation report for
  the full horizon-sweep numbers; this remains a smoke-scale finding (n=15
  instances, 1 seed) — full-scale confirmation is deferred to the PI's
  separate `full`/`full_nl` run.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from causaltemp_xai.benchmarks.mechanisms import lag_window as _lag_window


class CARLARecourse:
    """Stub: CARLA-style causal recourse generator.

    The real implementation should:

    1. Encode the SCM causal ordering to determine which variables are
       *actionable* (can be intervened on by the individual).
    2. Optimise only over the perturbation delta on actionable variables:

           delta_actionable = delta * actionable_mask
           cf = x + delta_actionable

    3. Minimise ``lam_pred * pred_loss(model(cf), target) + lam_prox * ||delta||^2``.
    4. Propagate the intervention through the causal graph to update
       non-actionable downstream variables consistently.

    Alternatively, a CVAE-based approach can be used to sample plausible
    recourses directly from the learned latent space.

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
        Proximity regularisation weight.
    lr : float
        Adam learning rate.
    n_steps : int
        Gradient-descent iterations.
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
    ) -> None:
        self.target_class = target_class
        self.actionable_mask = actionable_mask
        self.lam_pred = lam_pred
        self.lam_prox = lam_prox
        self.lr = lr
        self.n_steps = n_steps
        self.t0_fractions = t0_fractions

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

    def generate(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> np.ndarray:
        """Generate a causally-faithful recourse for instance *x*.

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

        Returns
        -------
        cf : ndarray of shape ``(T, k)``.

        Notes
        -----
        Only ``x[t0]`` (on actionable variables) is free; everything before
        ``t0`` is held equal to ``x`` (zero retroactive change) and everything
        after ``t0`` is the deterministic noiseless VAR rollout, so the CF is
        causally faithful (rollout-hard = 1) by construction. The intervention
        point ``t0`` is chosen from a small early candidate set; the best
        (flip achieved, lowest proximity) CF is returned.
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

        candidates = sorted({max(1, round(f * T)) for f in self.t0_fractions})
        candidates = [t0 for t0 in candidates if 0 < t0 < T - 1] or [T // 2]

        best = None  # (flipped, prox, cf_array)
        for t0 in candidates:
            delta = torch.zeros(k, requires_grad=True)
            optimiser = torch.optim.Adam([delta], lr=self.lr)
            x_orig_t0 = x_t[t0]

            for _step in range(self.n_steps):
                optimiser.zero_grad()
                x_t0 = x_orig_t0 + mask * delta
                x_cf = self._rollout(x_t, t0, x_t0, mechanism)
                logits = model.torch_logits(x_cf)
                pred_loss = F.cross_entropy(logits, target)
                prox = ((mask * delta) ** 2).sum()
                loss = self.lam_pred * pred_loss + self.lam_prox * prox
                loss.backward()
                optimiser.step()

            with torch.no_grad():
                x_t0 = x_orig_t0 + mask * delta
                x_cf = self._rollout(x_t, t0, x_t0, mechanism)
                logits = model.torch_logits(x_cf)
                flipped = int(logits.argmax(dim=1).item()) == self.target_class
                prox_val = float(((mask * delta) ** 2).sum().item())
                cf_arr = x_cf.detach().cpu().numpy().astype(np.float32)

            cand = (flipped, prox_val, cf_arr)
            # Prefer a flipping CF; among same flip-status, prefer lower proximity.
            if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):
                best = cand

        return best[2]

    # ------------------------------------------------------------------
    # CFExplainer alias interface + causal-info setter
    # ------------------------------------------------------------------

    def set_causal_info(self, graph, mechanism) -> None:
        """Store SCM graph and mechanism for use in fit/explain."""
        self._graph = graph
        self._mechanism = mechanism

    def fit(self, X_train, classifier) -> None:
        """No-op — CARLA needs graph/mechanism, set via set_causal_info()."""
        pass

    def explain(self, x, target_class: int, classifier) -> np.ndarray:
        """Alias for generate(x, classifier, self._graph, self._mechanism)."""
        graph = getattr(self, "_graph", None)
        mechanism = getattr(self, "_mechanism", None)
        if graph is None or mechanism is None:
            raise RuntimeError("Call set_causal_info(graph, mechanism) before explain().")
        return self.generate(x, classifier, graph, mechanism)

    def generate_batch(self, X: np.ndarray, model, graph=None, mechanism=None) -> np.ndarray:
        """Generate one CF per instance in ``X``."""
        X = np.asarray(X, dtype=np.float32)
        if graph is None:
            graph = getattr(self, "_graph", None)
        if mechanism is None:
            mechanism = getattr(self, "_mechanism", None)
        return np.stack([self.generate(x, model, graph, mechanism) for x in X], axis=0)


# =============================================================================
# Pearl-semantics variant (M2, 2026-07-08) — noise-reinjecting recourse
# =============================================================================


class PearlCARLARecourse:
    """Pearl-semantics causal recourse: noise-reinjecting counterfactual rollout.

    Same optimisation objective, actionability masking, and t0-candidate
    search as :class:`CARLARecourse`, but the forward rollout reinjects the
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

    **``lam_prox`` default differs from ``CARLARecourse`` (0.1 vs 0.5) — this
    is a deliberate, empirically-motivated choice, not an oversight.** The
    Pearl delta obeys the homogeneous recursion
    ``delta[t] = sum_l A_l @ delta[t-l]`` (factual noise cancels exactly), so
    under this benchmark's stability requirement (spectral radius < 1) any
    one-shot intervention decays geometrically and needs a proportionally
    larger ``delta[t0]`` to still matter at a long horizon. CARLA's default
    ``lam_prox=0.5`` quadratically over-penalizes that larger delta and,
    empirically (smoke-scale, 2026-07-08), turns a partial long-horizon
    validity problem into a near-total one (0.07 vs CARLA's 0.40 at the
    longest tested horizon). ``lam_prox=0.1`` closes most of that gap (see
    module docstring for the full horizon sweep) while ``pearl_hard=1`` is
    unaffected either way (it is a property of the rollout construction, not
    of the optimisation weights).

    Deliberately **not** a subclass of / refactor into :class:`CARLARecourse`:
    kept fully independent (duplicating the small optimisation-loop structure)
    so ``CARLARecourse``'s pinned behaviour — ``tests/test_methods.py::
    TestCARLA`` and the M1-regenerated ``results/`` ``rollout_hard=1`` rows —
    carries zero refactor risk. This mirrors the codebase's existing
    convention of small per-class duplication over shared-base abstraction
    for CF generators (see e.g. ``methods/counterfactual/cfts_methods.py``,
    where every ``Cfts*CF`` class repeats its own ``generate_batch``).

    Parameters mirror :class:`CARLARecourse` exactly (see its docstring)
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
    ) -> None:
        self.target_class = target_class
        self.actionable_mask = actionable_mask
        self.lam_pred = lam_pred
        self.lam_prox = lam_prox
        self.lr = lr
        self.n_steps = n_steps
        self.t0_fractions = t0_fractions

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
        — same structure as ``CARLARecourse._rollout`` but adds back the
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

    def generate(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanism,
    ) -> np.ndarray:
        """Generate a Pearl-faithful causal recourse for instance *x*.

        Same signature, actionability masking, t0-candidate search, and
        Adam-optimised objective as ``CARLARecourse.generate`` — only the
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

        candidates = sorted({max(1, round(f * T)) for f in self.t0_fractions})
        candidates = [t0 for t0 in candidates if 0 < t0 < T - 1] or [T // 2]

        best = None  # (flipped, prox, cf_array)
        for t0 in candidates:
            delta = torch.zeros(k, requires_grad=True)
            optimiser = torch.optim.Adam([delta], lr=self.lr)
            x_orig_t0 = x_t[t0]

            for _step in range(self.n_steps):
                optimiser.zero_grad()
                x_t0 = x_orig_t0 + mask * delta
                x_cf = self._rollout_pearl(x_t, t0, x_t0, mechanism, eps)
                logits = model.torch_logits(x_cf)
                pred_loss = F.cross_entropy(logits, target)
                prox = ((mask * delta) ** 2).sum()
                loss = self.lam_pred * pred_loss + self.lam_prox * prox
                loss.backward()
                optimiser.step()

            with torch.no_grad():
                x_t0 = x_orig_t0 + mask * delta
                x_cf = self._rollout_pearl(x_t, t0, x_t0, mechanism, eps)
                logits = model.torch_logits(x_cf)
                flipped = int(logits.argmax(dim=1).item()) == self.target_class
                prox_val = float(((mask * delta) ** 2).sum().item())
                cf_arr = x_cf.detach().cpu().numpy().astype(np.float32)

            cand = (flipped, prox_val, cf_arr)
            # Prefer a flipping CF; among same flip-status, prefer lower proximity.
            if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):
                best = cand

        return best[2]

    def generate_batch(
        self, X: np.ndarray, model, graph: np.ndarray = None, mechanism=None
    ) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        X = np.asarray(X, dtype=np.float32)
        if graph is None:
            graph = getattr(self, "_graph", None)
        if mechanism is None:
            mechanism = getattr(self, "_mechanism", None)
        return np.stack([self.generate(x, model, graph, mechanism) for x in X], axis=0)

    # ------------------------------------------------------------------
    # CFExplainer alias interface + causal-info setter (parity with CARLARecourse)
    # ------------------------------------------------------------------

    def set_causal_info(self, graph, mechanism) -> None:
        """Store SCM graph and mechanism for use in fit/explain."""
        self._graph = graph
        self._mechanism = mechanism

    def fit(self, X_train, classifier) -> None:
        """No-op — PearlCARLARecourse needs graph/mechanism, set via set_causal_info()."""
        pass

    def explain(self, x, target_class: int, classifier) -> np.ndarray:
        """Alias for generate(x, classifier, self._graph, self._mechanism)."""
        graph = getattr(self, "_graph", None)
        mechanism = getattr(self, "_mechanism", None)
        if graph is None or mechanism is None:
            raise RuntimeError("Call set_causal_info(graph, mechanism) before explain().")
        return self.generate(x, classifier, graph, mechanism)
