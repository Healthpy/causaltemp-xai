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

from typing import Optional

import numpy as np


def _window(arr: np.ndarray, t: int, L: int, k: int) -> np.ndarray:
    """Build the ``(L, k)`` lag window feeding the mechanism at time ``t``.

    Rows are ordered oldest→newest (``window[-1]`` is lag 1, ``x_{t-1}``); any
    row that would reach before ``t=0`` is left zero, replicating the original
    ``if lag_t >= 0`` guard (a zero row contributes nothing).
    """
    window = np.zeros((L, k))
    for j in range(L):
        src = t - L + j  # row position j maps to absolute time `src`
        if src >= 0:
            window[j] = arr[src]
    return window


class CFfaith:
    """Causal faithfulness scorer for temporal counterfactuals.

    Given the original instance ``x``, the proposed counterfactual ``x_cf``,
    the time step at which an intervention is claimed to occur
    (``intervention_t``), the lagged causal graph, and the VAR mechanisms, the
    scorer:

    1. **Retroactive check** – verifies that ``x_cf`` is identical to ``x``
       for all time steps *before* ``intervention_t``.  Any modification prior
       to the intervention is a retroactive causality violation.

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
    ) -> None:
        """Create a CF-faithfulness scorer.

        Parameters
        ----------
        tol:
            Residual threshold for the binary ``hard`` score and the retroactive
            change check.
        scale:
            Normalising scale in the ``soft`` score ``exp(-residual / scale)``.
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
            raise ValueError(
                f"semantics must be one of {self.SEMANTICS}, got {semantics!r}"
            )
        self.tol = tol
        self.scale = scale
        self.semantics = semantics

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
        x_orig = np.asarray(x_original, dtype=float)   # (T, k)
        x_cf_arr = np.asarray(x_cf, dtype=float)        # (T, k)
        T, k = x_orig.shape
        L = mechanism.L

        # ------------------------------------------------------------------
        # (i) Retroactive change check
        # ------------------------------------------------------------------
        retro_delta = np.abs(x_cf_arr[:intervention_t] - x_orig[:intervention_t]).sum()
        has_retroactive = retro_delta > self.tol

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
