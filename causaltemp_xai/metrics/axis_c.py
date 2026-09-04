"""Axis-C counterfactual evaluation metrics.

Implements the standard axes for evaluating counterfactual *quality* — i.e.
"does the counterfactual itself look good":

* **Validity**        – CF achieves the desired class under the black-box model.
* **Proximity**       – CF is close to the original instance (L1 or L2).
* **Sparsity**        – few features differ between the original and the CF.
* **OOD Plausibility** – CF lies within the training distribution, estimated
                          via sklearn's IsolationForest.
* **TRSI**            – mechanism-free descriptor of how smoothly the edit
                          varies over time (see :func:`trsi` for its scope —
                          it is a proxy, not a faithfulness criterion).

Whether the CF is *causally faithful* — consistent with the data-generating
mechanism, including the absence of retroactive pre-intervention edits — is a
separate question answered by
:class:`~causaltemp_xai.metrics.cf_faith.CFfaith`, not by this module.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.ensemble import IsolationForest

# Single source of truth for "is this element changed?" across the benchmark
# (shared with derive_intervention_t and CFfaith's retro gate) — sparsity uses
# the same predicate so one benchmark has one definition of "changed"
# (metric-quality fix #4, 2026-07-18; previously an absolute 1e-6 that pinned
# gradient methods at sparsity 0.00).
from causaltemp_xai.scm.intervention import INTERVENTION_TOL

# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def validity(
    x_cf: np.ndarray,
    model,
    target_class: int,
) -> float:
    """Fraction of counterfactuals predicted as ``target_class``.

    This is the standard CF *validity* (flip-rate): the share of proposed
    counterfactuals that the black-box model actually assigns to the desired
    class.  A value of ``1.0`` means every CF achieved the target.

    Parameters
    ----------
    x_cf:
        Counterfactual instance(s).  Either a single instance of shape
        ``(T, k)`` or a batch of shape ``(N, T, k)``.  A batch dimension is
        added automatically for a single ``(T, k)`` instance.
    model:
        Classifier.  Either an object exposing ``predict`` (e.g.
        :class:`~causaltemp_xai.classifiers.LSTMClassifier`) or a plain callable
        mapping a ``(N, T, k)`` batch to integer labels of shape ``(N,)``.
    target_class:
        The desired output class the counterfactuals should achieve.

    Returns
    -------
    float
        Validity (flip-rate) in ``[0, 1]``.
    """
    predict = getattr(model, "predict", model)
    batch = np.asarray(x_cf, dtype=float)
    if batch.ndim == 2:  # single (T, k) → add batch dim
        batch = batch[np.newaxis]
    preds = np.asarray(predict(batch)).reshape(-1)
    return float(np.mean(preds == target_class))


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------


def proximity(
    x_original: np.ndarray,
    x_cf: np.ndarray,
    norm: Literal["l1", "l2"] = "l1",
) -> float:
    """Distance between the original instance and the counterfactual.

    Parameters
    ----------
    x_original:
        Original time series, shape ``(T, k)`` or ``(k,)``.
    x_cf:
        Counterfactual, same shape as ``x_original``.
    norm:
        ``"l1"`` (Manhattan) or ``"l2"`` (Euclidean).

    Returns
    -------
    float
        Scalar distance.  Lower is closer (better).
    """
    x_orig = np.asarray(x_original, dtype=float).ravel()
    x_cf_arr = np.asarray(x_cf, dtype=float).ravel()
    diff = x_orig - x_cf_arr
    if norm == "l1":
        return float(np.sum(np.abs(diff)))
    elif norm == "l2":
        return float(np.sqrt(np.sum(diff**2)))
    else:
        raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------


def sparsity(
    x_original: np.ndarray,
    x_cf: np.ndarray,
    tol: float = INTERVENTION_TOL,
    return_detailed: bool = False,
) -> float | dict[str, float]:
    """Fraction of features that are unchanged between original and CF.

    A higher sparsity score (closer to 1) means fewer features were modified,
    which is generally desirable for interpretability.

    For temporal data (shape ``(T, k)``), two complementary sparsity metrics
    are available:
    - **channel_sparsity**: Fraction of channels entirely unchanged across time.
    - **timepoint_sparsity**: Fraction of timepoints entirely unchanged across channels.

    Parameters
    ----------
    x_original:
        Original instance, shape ``(T, k)`` or ``(k,)``.
    x_cf:
        Counterfactual instance, same shape.
    tol:
        Per-element threshold below which a difference counts as unchanged.
        Defaults to :data:`~causaltemp_xai.scm.intervention.INTERVENTION_TOL`
        — the benchmark's single definition of "changed", shared with
        ``derive_intervention_t`` and CF-faith's retroactive gate (fix #4,
        2026-07-18). The old absolute ``1e-6`` counted float-scale gradient
        perturbations as edits, pinning gradient methods at 0.00.
    return_detailed:
        If False (default), return scalar sparsity (all features flattened).
        If True and input is temporal ``(T, k)``, return dict with
        ``{"channels": float, "timepoints": float}``.

    Returns
    -------
    float or dict
        If ``return_detailed=False``: scalar in ``[0, 1]``.
        If ``return_detailed=True``: dict with "channels" and "timepoints" keys.
    """
    x_orig = np.asarray(x_original, dtype=float)
    x_cf_arr = np.asarray(x_cf, dtype=float)

    if not return_detailed or x_orig.ndim != 2:
        # Return scalar (flattened sparsity)
        diff_flat = x_orig.ravel() - x_cf_arr.ravel()
        n_unchanged = int(np.sum(np.abs(diff_flat) <= tol))
        return n_unchanged / len(diff_flat)

    # Temporal data: compute channel and timepoint sparsity
    T, k = x_orig.shape
    diff = np.abs(x_orig - x_cf_arr)

    # Channel sparsity: fraction of channels entirely unchanged across all timepoints
    channels_unchanged = np.sum(np.all(diff <= tol, axis=0))  # which channels: all T unchanged?
    channel_sparsity = channels_unchanged / k

    # Timepoint sparsity: fraction of timepoints entirely unchanged across all channels
    timepoints_unchanged = np.sum(np.all(diff <= tol, axis=1))  # which timepoints: all k unchanged?
    timepoint_sparsity = timepoints_unchanged / T

    return {
        "channels": channel_sparsity,
        "timepoints": timepoint_sparsity,
    }


# ---------------------------------------------------------------------------
# OOD Plausibility
# ---------------------------------------------------------------------------


def ood_plausibility(
    x_train: np.ndarray,
    x_cf: np.ndarray,
    method: Literal["if"] = "if",
    contamination: float = 0.05,
    random_state: int = 0,
) -> float:
    """Estimate whether the counterfactual lies within the training distribution.

    Uses an IsolationForest trained on ``x_train`` to score ``x_cf``.
    The IsolationForest ``decision_function`` returns higher values for
    in-distribution points; we return that score directly so callers can
    interpret it (positive = plausible, negative = anomalous).

    Parameters
    ----------
    x_train:
        Training instances, shape ``(N, T, k)`` or ``(N, k)``.  Used to fit
        the IsolationForest.
    x_cf:
        Counterfactual instance to evaluate, shape ``(T, k)`` or ``(k,)``.
        A batch of CFs of shape ``(M, T, k)`` is also accepted.
    method:
        Currently only ``"if"`` (IsolationForest) is supported.
    contamination:
        Expected fraction of outliers in the training set.  Passed directly
        to :class:`sklearn.ensemble.IsolationForest`.
    random_state:
        Random seed for IsolationForest reproducibility.

    Returns
    -------
    float or ndarray
        IsolationForest anomaly score(s).  Higher = more plausible (in-dist).
        Returns a scalar for a single CF, or an array for a batch.
    """
    if method != "if":
        raise ValueError(f"method must be 'if', got {method!r}")

    X_tr = np.asarray(x_train, dtype=float)
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X_tr_flat)

    x_cf_arr = np.asarray(x_cf, dtype=float)
    if x_cf_arr.ndim == X_tr.ndim - 1:
        # Single instance
        x_cf_flat = x_cf_arr.ravel().reshape(1, -1)
        scores = clf.decision_function(x_cf_flat)
        return float(scores[0])
    else:
        # Batch
        x_cf_flat = x_cf_arr.reshape(x_cf_arr.shape[0], -1)
        return clf.decision_function(x_cf_flat)


# ---------------------------------------------------------------------------
# SCM-noise plausibility — ground-truth-exact plausibility (fix #8, 2026-07-18)
# ---------------------------------------------------------------------------


def scm_noise_plausibility(
    x_cf: np.ndarray,
    mechanism,
    noise_scale: float,
    t_start: int | None = None,
) -> float:
    """Ground-truth plausibility: is the CF's implied noise calibrated to the SCM's?

    The benchmark owns the data-generating process, so plausibility need not
    be estimated (IsolationForest on flattened trajectories — see
    :func:`ood_plausibility`, now the mechanism-free stand-in): abduct the
    noise the CF *implies* under the true mechanism,
    ``eps_hat[t] = x_cf[t] - f(window_t)`` (exact under additive noise), and
    compare its empirical scale to the SCM's true noise scale ``b``
    (for Laplace(0, b) noise, ``E|eps| = b``):

    .. math::

        \\mathrm{plaus} = \\exp\\big(-\\,\\big|\\ln(\\hat{s}/b)\\big|\\big),
        \\qquad \\hat{s} = \\mathrm{mean}\\,|\\hat{\\varepsilon}|

    * ``1.0`` — the CF's innovations are exactly noise-calibrated
      (an on-manifold trajectory, e.g. the Pearl oracle).
    * ``→ 0`` as the implied noise is far too **large** (arbitrary edits the
      mechanism cannot absorb) or far too **small** (a noiseless skeleton:
      ``eps ≡ 0`` post-``t0`` is maximally *likely* pointwise but maximally
      implausible as a draw from the noise law — the log-ratio catches what a
      naive likelihood would reward).

    Dimension-free, calibrated, temporal by construction — no estimator, no
    contamination hyperparameter. Residuals at ``t < L`` are excluded (they
    absorb initial conditions; see ``abduct_noise``).

    Parameters
    ----------
    x_cf:
        Counterfactual, shape ``(T, k)`` or batch ``(N, T, k)``.
    mechanism:
        The true :class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism`.
    noise_scale:
        The SCM's true noise scale ``b`` (``BenchmarkConfig.noise_scale``).
    t_start:
        First timestep whose residual is scored. Defaults to ``mechanism.L``.
        Pass the intervention step to score only the post-intervention region.

    Returns
    -------
    float in (0, 1] — mean over the batch for batched input.
    """
    from causaltemp_xai.benchmarks.structural_cf import abduct_noise

    x = np.asarray(x_cf, dtype=float)
    if x.ndim == 2:
        x = x[np.newaxis]
    lo = mechanism.L if t_start is None else int(t_start)
    scores = []
    for inst in x:
        eps_hat = abduct_noise(inst, mechanism)[lo:]
        s_hat = float(np.abs(eps_hat).mean())
        ratio = (s_hat + 1e-12) / float(noise_scale)
        scores.append(float(np.exp(-abs(np.log(ratio)))))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# TRSI — temporal smoothness of the edit (bench-ported, mechanism-free proxy)
# ---------------------------------------------------------------------------


def trsi(X_cf: np.ndarray, X: np.ndarray) -> float:
    """Temporal Relevance Smoothness Index (TRSI) — a *mechanism-free* descriptor
    of how abruptly a counterfactual's edit varies over time.

    .. math::

        \\mathrm{TRSI} = \\frac{1}{T-1} \\sum_t
            \\lVert \\Delta_{t+1} - \\Delta_t \\rVert_2,
        \\qquad \\Delta_t = X^{cf}_t - X_t

    The L2 norm is over channels ``k``, then averaged over ``T-1`` steps and
    ``N`` instances. Low TRSI ⇒ the edit varies smoothly; high TRSI ⇒ the edit
    jitters from step to step.

    Scope and interpretation (read before reporting)
    ------------------------------------------------
    TRSI measures the smoothness of the **perturbation** ``Δ``, not of the
    trajectory, and it is *not* a causal-faithfulness metric:

    * **What it adds.** Proximity measures *how much* changed and sparsity *how
      many* features changed; neither notices that a CF bought a small, sparse
      edit by injecting temporally incoherent noise. TRSI is the axis that
      catches that.
    * **What it is not.** TRSI is a **heuristic proxy**, not a principled
      criterion, and it carries no ground-truth grounding: it never consults the
      SCM, so it cannot certify that an edit is one the mechanism could have
      produced. :class:`~causaltemp_xai.metrics.cf_faith.CFfaith` answers that
      question directly and strictly better. A mechanism-consistent CF will
      generally be TRSI-smooth, but the converse does **not** hold — a smooth
      edit can still be causally impossible.
    * **Why keep it.** It requires no mechanism, so unlike CF-faith it transfers
      to real datasets where no SCM is available. Report it as the mechanism-free
      stand-in for CF-faith and a complementary edit-quality descriptor —
      never as evidence of causal faithfulness.
    * **Direction.** Lower is smoother. TRSI is a descriptor with no
      ground-truth optimum: a *genuinely* abrupt intervention should produce a
      high TRSI, so it must be read against the other Axis-C columns rather than
      minimised on its own.

    Ported from ``causal_tscf_bench/metrics/axis_c.py``; the formulation is
    adopted, not novel to this benchmark.

    Normalization (metric-quality fix #5, 2026-07-18)
    -------------------------------------------------
    TRSI is reported as **relative jitter**: the mean step-to-step change of
    the edit divided by the edit's own mean magnitude,

    .. math::

        \\mathrm{TRSI} = \\frac{\\tfrac{1}{T-1}\\sum_t
            \\lVert \\Delta_{t+1} - \\Delta_t \\rVert_2}
            {\\tfrac{1}{T}\\sum_t \\lVert \\Delta_t \\rVert_2 + \\epsilon}.

    The unnormalized form scaled with the edit's magnitude, partially
    re-measuring proximity rather than temporal coherence. The ratio is
    dimensionless and scale-invariant: doubling the edit leaves it unchanged.
    Caveat retained from the original: a genuine step intervention at ``t0``
    contributes one legitimate large first-difference (the onset), so TRSI
    must still be read against the other Axis-C columns, never minimised
    on its own.

    Parameters
    ----------
    X_cf : (N, T, k) or (T, k) -- counterfactuals
    X    : (N, T, k) or (T, k) -- factuals

    Returns
    -------
    float
        Relative jitter of ``Δ``. Non-negative; 0 for a constant (perfectly
        smooth) edit; 0 for an identical CF (zero edit) by convention.
    """
    X_cf_arr = np.asarray(X_cf, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if X_cf_arr.ndim == 2:
        X_cf_arr = X_cf_arr[np.newaxis]
        X_arr = X_arr[np.newaxis]
    delta = X_cf_arr - X_arr  # (N, T, k)
    d_delta = np.diff(delta, axis=1)  # (N, T-1, k) -- first difference
    l2_per_step = np.sqrt((d_delta**2).sum(axis=-1))  # (N, T-1)
    l2_delta = np.sqrt((delta**2).sum(axis=-1))  # (N, T) -- edit magnitude
    denom = float(l2_delta.mean())
    if denom < 1e-12:
        return 0.0  # zero edit: nothing to be jittery about
    return float(l2_per_step.mean() / denom)


def compute_axis_c(
    X: np.ndarray,
    X_cf_exp: np.ndarray,
    X_train: np.ndarray,
    classifier,
    target_class: int,
    mechanism=None,
    noise_scale: float | None = None,
) -> dict:
    """Compute all Axis C metrics.

    The retroactive-edit check that IVR used to provide lives in
    :class:`~causaltemp_xai.metrics.cf_faith.CFfaith`, whose retro gate zeroes
    both scores on any pre-intervention edit — see the CF-faith docstring.

    Parameters
    ----------
    X          : (N, T, k) factual instances
    X_cf_exp   : (N, T, k) explainer CFs
    X_train    : (N_train, T, k) training distribution
    classifier : classifier with .predict()
    target_class : int
    mechanism  : optional Mechanism — with ``noise_scale``, adds the
                 ground-truth ``scm_noise_plausibility`` column (fix #8)
    noise_scale : optional float — the SCM's true noise scale

    Returns
    -------
    dict keyed exactly as the pipeline's result files are — ``validity``,
    ``proximity_l1``, ``proximity_l2``, ``sparsity``, ``ood``, ``trsi``, and
    (given mechanism + noise_scale) ``scm_noise_plausibility``.

    **The keys are snake_case deliberately.** They were TitleCase
    (``Validity``, ``SCM_Plausibility``, …) until 2026-08-03, which made this a
    second naming vocabulary for metrics the production path already emitted —
    so anything wiring this function in would have written keys no axis claims.
    One metric, one name (``metrics/taxonomy.py``).
    """
    results: dict = {}
    results["validity"] = validity(X_cf_exp, classifier, target_class)
    results["proximity_l1"] = float(
        np.mean([proximity(X[i], X_cf_exp[i], norm="l1") for i in range(len(X))])
    )
    results["proximity_l2"] = float(
        np.mean([proximity(X[i], X_cf_exp[i], norm="l2") for i in range(len(X))])
    )
    results["sparsity"] = float(np.mean([sparsity(X[i], X_cf_exp[i]) for i in range(len(X))]))
    results["trsi"] = trsi(X_cf_exp, X)

    ood_scores = np.atleast_1d(ood_plausibility(X_train, X_cf_exp))
    results["ood"] = float(np.mean(ood_scores))

    if mechanism is not None and noise_scale is not None:
        results["scm_noise_plausibility"] = scm_noise_plausibility(X_cf_exp, mechanism, noise_scale)

    return results
