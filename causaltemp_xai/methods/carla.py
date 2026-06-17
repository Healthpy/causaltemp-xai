"""CARLA-style causal recourse — stub.

Reference
---------
Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter, T., & Kasneci, G.
(2021). *CARLA: A Python Library to Benchmark Algorithmic Recourse and
Counterfactual Explanation Algorithms.*  NeurIPS 2021 Datasets and Benchmarks.

This stub defines the ``CARLARecourse`` class interface.  The real
implementation would train a generative model (e.g. CVAE) or apply
constrained optimisation that respects actionability constraints and the
causal ordering over variables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


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
    def _rollout(x_t: torch.Tensor, t0: int, x_t0: torch.Tensor, A_list) -> torch.Tensor:
        """Build the CF trajectory: original up to ``t0``, free values at ``t0``,
        noiseless VAR rollout after ``t0`` (differentiable w.r.t. ``x_t0``).

        ``x_cf[t] = sum_l A_l @ x_cf[t-l-1]`` for ``t > t0`` (per-sample ``A @ x``,
        matching ``cf_faith.py``). Never re-injects noise → the CF *is* its own
        noiseless rollout, so CFfaith(noiseless_rollout).hard == 1 by construction.
        """
        T = x_t.shape[0]
        rows: list[torch.Tensor] = []
        for t in range(T):
            if t < t0:
                rows.append(x_t[t])          # fixed original (zero retroactive change)
            elif t == t0:
                rows.append(x_t0)            # free / intervened values
            else:
                acc = torch.zeros_like(x_t0)
                for l, A in enumerate(A_list):
                    lag = t - l - 1
                    if lag >= 0:
                        acc = acc + A @ rows[lag]
                rows.append(acc)
        return torch.stack(rows, dim=0)      # (T, k)

    def generate(
        self,
        x: np.ndarray,
        model,
        graph: np.ndarray,
        mechanisms: list,
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
        mechanisms : list of ``(k, k)`` arrays
            VAR coefficient matrices ``A_l``.

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
        A_list = [torch.as_tensor(np.asarray(A, dtype=np.float32)) for A in mechanisms]
        target = torch.tensor([self.target_class], dtype=torch.long)

        if self.actionable_mask is None:
            mask = torch.ones(k, dtype=torch.float32)
        else:
            m = np.asarray(self.actionable_mask, dtype=np.float32)
            mask = torch.as_tensor(m[-1] if m.ndim == 2 else m)

        candidates = sorted({max(1, int(round(f * T))) for f in self.t0_fractions})
        candidates = [t0 for t0 in candidates if 0 < t0 < T - 1] or [T // 2]

        best = None  # (flipped, prox, cf_array)
        for t0 in candidates:
            delta = torch.zeros(k, requires_grad=True)
            optimiser = torch.optim.Adam([delta], lr=self.lr)
            x_orig_t0 = x_t[t0]

            for _step in range(self.n_steps):
                optimiser.zero_grad()
                x_t0 = x_orig_t0 + mask * delta
                x_cf = self._rollout(x_t, t0, x_t0, A_list)
                logits = model.torch_logits(x_cf)
                pred_loss = F.cross_entropy(logits, target)
                prox = ((mask * delta) ** 2).sum()
                loss = self.lam_pred * pred_loss + self.lam_prox * prox
                loss.backward()
                optimiser.step()

            with torch.no_grad():
                x_t0 = x_orig_t0 + mask * delta
                x_cf = self._rollout(x_t, t0, x_t0, A_list)
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
        self, X: np.ndarray, model, graph: np.ndarray, mechanisms: list
    ) -> np.ndarray:
        """Generate one CF per instance in ``X`` of shape ``(N, T, k)``."""
        X = np.asarray(X, dtype=np.float32)
        return np.stack(
            [self.generate(x, model, graph, mechanisms) for x in X], axis=0
        )
