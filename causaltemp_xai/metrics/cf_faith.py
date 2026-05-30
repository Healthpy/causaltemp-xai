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
            * ``"pearl_delta"`` — a CF is faithful iff the difference
              ``delta = x_cf - x_orig`` follows the homogeneous recursion
              ``delta[t] = sum_l A_l @ delta[t-l]`` for ``t > t0`` (the original
              innovations cancel via abduction). Rewards CFs that differ from the
              factual *only* by the propagated intervention (on-manifold; the
              textbook Pearl counterfactual).

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
        mechanisms: list[np.ndarray],
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
        mechanisms:
            List of *L* coefficient matrices ``A_l``, each of shape ``(k, k)``.
            ``x_t = sum_l A_l @ x_{t-l-1} + noise``.

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
        L = len(mechanisms)

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
        # The forward check homogeneously rolls a *target* trajectory forward
        # via the mechanisms and measures how far the proposed values deviate.
        # The only difference between the two semantics is which trajectory is
        # rolled/compared:
        #   * noiseless_rollout: the CF itself (faithful => CF is a noiseless
        #     SCM continuation of itself).
        #   * pearl_delta: the difference delta = x_cf - x_orig (faithful =>
        #     delta follows the homogeneous recursion; original noise cancels).
        if self.semantics == "noiseless_rollout":
            target = x_cf_arr
        else:  # "pearl_delta"
            target = x_cf_arr - x_orig

        simulated = target.copy()  # values at <= intervention_t are held fixed

        # Re-simulate time steps *after* intervention_t using the mechanisms
        for t in range(intervention_t + 1, T):
            # Predict x_t from the history of simulated (post-intervention) values
            x_t_pred = np.zeros(k)
            for l, A in enumerate(mechanisms):
                lag_t = t - l - 1
                if lag_t >= 0:
                    x_t_pred += A @ simulated[lag_t]
            simulated[t] = x_t_pred

        # L1 residual between the forward-simulated trajectory and the target
        residual_region = target[intervention_t:]
        simulated_region = simulated[intervention_t:]
        l1_residual = float(np.abs(residual_region - simulated_region).mean())

        hard = float(l1_residual < self.tol)
        soft = float(np.exp(-l1_residual / (self.scale + 1e-12)))

        return {"hard": hard, "soft": soft}
