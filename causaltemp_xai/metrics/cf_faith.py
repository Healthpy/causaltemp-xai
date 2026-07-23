"""Causal Faithfulness (CF-faith) metrics for temporal counterfactuals.

A counterfactual explanation is *causally faithful* if it respects the
data-generating SCM: changes introduced before the claimed intervention
timestep are penalised (retroactive causality violation), and downstream
variables should evolve according to the known VAR mechanisms rather than
being set arbitrarily.

The ``CFfaith`` class exposes a ``score()`` method that returns both a hard
(binary) and a soft (continuous) faithfulness score.
"""

from __future__ import annotations

import numpy as np

# Single source of truth for the lag-window contract (see mechanisms module
# docstring). Imported here so cf_faith and structural_cf cannot drift.
from causaltemp_xai.benchmarks.mechanisms import lag_window as _window

# Single source of truth for the per-element "is this element changed?"
# threshold, shared with derive_intervention_t (M1 decision, 2026-07-07) — see
# the constant's docstring for the full rationale. The retroactive gate below
# is now the benchmark's only consumer of it (axis_c.ivr was retired
# 2026-07-15; see docs/spec_code_reconciliation.md §4.1).
from causaltemp_xai.scm.intervention import INTERVENTION_TOL


class CFfaith:
    """Causal faithfulness scorer for temporal counterfactuals.

    Given the original instance ``x``, the proposed counterfactual ``x_cf``,
    the time step at which an intervention is claimed to occur
    (``intervention_t``), the lagged causal graph, and the VAR mechanisms, the
    scorer:

    1. **Retroactive check** – verifies that ``x_cf`` is identical to ``x``
       for all time steps *before* ``intervention_t``.  Any modification prior
       to the intervention is a retroactive causality violation.  "Modified"
       uses the same per-element predicate (max ``|Δ| > retro_tol``, default
       :data:`~causaltemp_xai.scm.intervention.INTERVENTION_TOL`) that
       ``derive_intervention_t`` uses to *define* the intervention timestep,
       so a region certified unchanged by the t0 definition can never be
       flagged retroactive by this gate (M1 fix, 2026-07-07 — previously a
       *summed* L1 vs ``1e-4`` accumulated float-scale reconstruction noise
       over the T×k pre-window and falsely flagged full-trajectory-
       reconstruction methods such as CELS).

    2. **SCM forward simulation** – starting from ``intervention_t``, simulates
       the SCM forward using the known VAR coefficient matrices and computes
       the L1 residual between the simulated trajectory and the proposed CF.
       A small residual means the CF is consistent with the causal mechanisms.

    Scores
    ------
    * **hard** – 1 if no retroactive changes *and* the simulated residual is
      below ``tol``; 0 otherwise.
    * **soft** – ``exp(-residual / scale)`` in ``[0, 1]``, where ``scale``
      normalises by the number of features; always 0 when retroactive changes
      are present.
    """

    #: Supported faithfulness semantics (see :meth:`score`).
    SEMANTICS = ("noiseless_rollout", "pearl_delta")

    def __init__(
        self,
        tol: float = 1e-4,
        scale: float = 1.0,
        semantics: str = "noiseless_rollout",
        retro_tol: float = INTERVENTION_TOL,
    ) -> None:
        """Create a CF-faithfulness scorer.

        Parameters
        ----------
        tol:
            Residual threshold for the binary ``hard`` score (forward
            SCM-consistency check only).
        scale:
            Normalising scale in the ``soft`` score ``exp(-residual / scale)``.
        retro_tol:
            Per-element threshold for the retroactive change check. Defaults
            to :data:`~causaltemp_xai.scm.intervention.INTERVENTION_TOL` so
            the gate is consistent, by construction, with the
            ``derive_intervention_t`` heuristic that defines ``intervention_t``
            in the evaluation pipeline. Kept separate from ``tol``: the
            forward residual is a *mean* over the post-intervention region
            (a different quantity at a different scale), while the retro gate
            is a per-element edit detector.
        semantics:
            Which faithfulness definition the forward check uses:

            * ``"noiseless_rollout"`` (default) — a CF is faithful iff
              ``x_cf[t0:]`` *is* the deterministic, noiseless VAR rollout of
              itself. Rewards trajectories that are pure SCM continuations
              (off the noisy data manifold).
            * ``"pearl_delta"`` — a CF is faithful iff it equals the
              **abduction-action-prediction** counterfactual: abduct the
              factual exogenous noise ``eps[t] = x_orig[t] -
              mechanism.forward_numpy(window)`` (exact under additive noise),
              hold the pre-intervention prefix + intervened step, then roll
              forward reusing ``eps`` (``x_pred[t] = mechanism.forward_numpy(
              window) + eps[t]``). Rewards CFs that differ from the factual
              *only* by the propagated intervention (on-manifold; the textbook
              Pearl counterfactual). For a **linear** mechanism this reduces
              exactly to the homogeneous recursion ``delta[t] = sum_l A_l @
              delta[t-l]`` (the factual innovations cancel), so the v0.1 linear
              numbers are unchanged; for a nonlinear mechanism it is the correct
              generalization (linear superposition no longer holds).

            The two reward structurally different counterfactuals — a single CF
            cannot score ``hard=1`` under both. See the project plan's index
            "Decisions" (keep both CF-faith metrics).
        """
        if semantics not in self.SEMANTICS:
            raise ValueError(f"semantics must be one of {self.SEMANTICS}, got {semantics!r}")
        self.tol = tol
        self.scale = scale
        self.semantics = semantics
        self.retro_tol = retro_tol

    def score(
        self,
        x_original: np.ndarray,
        x_cf: np.ndarray,
        intervention_t: int,
        graph: np.ndarray,
        mechanism,
    ) -> dict[str, float]:
        """Compute hard and soft CF-faithfulness scores.

        Parameters
        ----------
        x_original:
            Original time series, shape ``(T, k)``.
        x_cf:
            Proposed counterfactual, shape ``(T, k)``.
        intervention_t:
            Index of the first time step where the intervention is applied.
            Time steps ``0 … intervention_t-1`` must be unchanged.
        graph:
            Binary adjacency tensor of shape ``(k, k, L)`` from the SCM.
            Not used directly in computation but kept for API completeness /
            downstream analysis.
        mechanism:
            A :class:`~causaltemp_xai.benchmark.mechanisms.Mechanism` producing
            the deterministic next-step mean ``x_t = mechanism.forward_numpy(window)``
            (for the linear case ``x_t = sum_l A_l @ x_{t-l-1}``).

        Returns
        -------
        dict with keys:

        ``"hard"``
            1.0 if no retroactive change and SCM residual < ``tol``, else 0.0.
        ``"soft"``
            Continuous score in ``[0, 1]``; 0.0 when retroactive changes exist.
        """
        x_orig = np.asarray(x_original, dtype=float)  # (T, k)
        x_cf_arr = np.asarray(x_cf, dtype=float)  # (T, k)
        T, k = x_orig.shape
        L = mechanism.L

        # ------------------------------------------------------------------
        # (i) Retroactive change check — per-element max, same predicate and
        # scale as derive_intervention_t (M1 decision, 2026-07-07). A summed
        # check here would grow with T×k from float-scale reconstruction
        # noise alone and contradict the very definition of intervention_t.
        # ------------------------------------------------------------------
        retro_region = np.abs(x_cf_arr[:intervention_t] - x_orig[:intervention_t])
        has_retroactive = retro_region.size > 0 and float(retro_region.max()) > self.retro_tol

        if has_retroactive:
            return {"hard": 0.0, "soft": 0.0}

        # ------------------------------------------------------------------
        # (ii) SCM forward simulation from intervention_t
        # ------------------------------------------------------------------
        # Each semantics builds a *reference* trajectory by rolling the
        # mechanism forward from intervention_t and measures how far the
        # proposed CF deviates from it. Both feed the mechanism exactly L rows
        # (oldest→newest), zero-padding rows that reach before t=0 — replicating
        # the original ``if lag_t >= 0`` guard (a zero row contributes nothing).
        if self.semantics == "noiseless_rollout":
            # Faithful iff x_cf[t0:] *is* the deterministic, noiseless rollout of
            # itself: roll x_cf forward with no noise and compare to x_cf.
            target = x_cf_arr
            simulated = target.copy()  # values at <= intervention_t held fixed
            for t in range(intervention_t + 1, T):
                window = _window(simulated, t, L, k)
                simulated[t] = mechanism.forward_numpy(window)
        else:  # "pearl_delta": abduction-action-prediction (Pearl's 3 steps)
            # 1. Abduct the factual exogenous noise (exact under additive noise).
            # 2. Action: hold x_cf[:t0+1] (prefix + intervened step).
            # 3. Predict: roll forward reusing the abducted noise.
            # For linear f this reduces to delta[t] = sum_l A_l @ delta[t-l]
            # (factual noise cancels); for nonlinear f it is the correct
            # generalization. Faithful iff x_cf[t0:] matches this prediction.
            target = x_cf_arr
            simulated = x_cf_arr.copy()  # x_pred; values at <= t0 held fixed
            for t in range(intervention_t + 1, T):
                eps_t = x_orig[t] - mechanism.forward_numpy(_window(x_orig, t, L, k))
                window = _window(simulated, t, L, k)
                simulated[t] = mechanism.forward_numpy(window) + eps_t

        # L1 residual between the reference trajectory and the proposed CF.
        residual_region = target[intervention_t:]
        simulated_region = simulated[intervention_t:]
        l1_residual = float(np.abs(residual_region - simulated_region).mean())

        hard = float(l1_residual < self.tol)
        soft = float(np.exp(-l1_residual / (self.scale + 1e-12)))

        return {"hard": hard, "soft": soft}
