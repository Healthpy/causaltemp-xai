"""Axis-A concept quality metrics.

Four metrics that jointly characterise the quality of a latent concept
representation produced by a concept-based XAI method (TCAV-T, CBM-T, iVAE,
β-VAE, LEAP, …) evaluated against ground-truth generative factors.

* **icc_latent** (model-based latent intervention) — Interventional Concept Consistency.
  Perturbs each latent dimension independently and counts how often the
  classifier changes its prediction.  Under Hyvärinen (2019) / Song (2024)
  nonlinear-ICA identifiability, a high ICC_i implies that concept i is
  causally relevant to the classifier's decision up to permutation and
  element-wise reparameterisation.

* **MIG** — Mutual Information Gap (Chen et al., 2018, β-TCVAE).
  For each true factor the gap between the two highest mutual-information
  latent dimensions is normalised by the factor's marginal entropy.
  A MIG of 1 means each factor is captured by a single latent; 0 means
  information is spread uniformly.

* **DCI-D / DCI-C** — Disentanglement and Completeness (Eastwood &
  Williams, 2018).  A random-forest regressor is trained to predict each
  true factor from inferred latents; its feature-importance matrix drives
  both scores.

* **MCC** — Mean Correlation Coefficient.
  Optimal linear matching between inferred and true latent dimensions via
  the Hungarian algorithm on the absolute Pearson correlation matrix.
  Standard identifiability benchmark metric.

References
----------
Chen et al. (2018). "Isolating Sources of Disentanglement in VAEs." NeurIPS.
Eastwood & Williams (2018). "A Framework for the Quantitative Evaluation
    of Disentangled Representations." ICLR.
Hyvärinen & Morioka (2019). "Nonlinear ICA Using Auxiliary Variables."
    AISTATS.
Song et al. (2024). "Identifiability of Sparse Causal Representations."
    NeurIPS.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_rows(M: np.ndarray) -> np.ndarray:
    """Divide each row by its sum; rows that sum to 0 remain 0."""
    row_sums = M.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, M / row_sums, 0.0)


def _normalise_cols(M: np.ndarray) -> np.ndarray:
    """Divide each column by its sum; columns that sum to 0 remain 0."""
    col_sums = M.sum(axis=0, keepdims=True)
    return np.where(col_sums > 0, M / col_sums, 0.0)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a normalised probability vector (nats)."""
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _mi_binned(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Empirical mutual information (nats) estimated via joint histogram."""
    N = len(x)
    x_edges = np.linspace(x.min(), x.max() + 1e-10, n_bins + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-10, n_bins + 1)
    x_disc = np.searchsorted(x_edges[1:-1], x)
    y_disc = np.searchsorted(y_edges[1:-1], y)

    # Build joint distribution
    p_xy = np.zeros((n_bins, n_bins))
    np.add.at(p_xy, (x_disc, y_disc), 1.0)
    p_xy /= N

    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    mask = p_xy > 0
    mi = float(np.sum(p_xy[mask] * np.log(p_xy[mask] / (p_x * p_y + 1e-12)[mask])))
    return max(mi, 0.0)


def _marginal_entropy_binned(x: np.ndarray, n_bins: int = 20) -> float:
    """Marginal entropy (nats) of a continuous variable via histogram."""
    counts, _ = np.histogram(x, bins=n_bins)
    p = counts / counts.sum()
    return _entropy(p)


# ---------------------------------------------------------------------------
# ICC — Interventional Concept Consistency
# ---------------------------------------------------------------------------


def icc_latent(
    X: np.ndarray,
    encoder: Callable,
    decoder: Callable,
    classifier,
    delta: Union[float, np.ndarray] = 1.0,
    scale_by_std: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """Interventional Concept Consistency (ICC) — matched-baseline latent traversal.

    For each latent dimension *i*, ICC measures how often a traversal along that
    dimension changes the classifier's prediction **relative to the unperturbed
    reconstruction** (not the original input):

    .. math::

        \\mathrm{ICC}_i = \\frac{1}{N} \\sum_{n}
            \\mathbb{1}\\!\\left[
                f\\!\\left(D_\\psi\\!\\left(z^n + \\delta_i e_i\\right)\\right)
                \\neq f\\!\\left(D_\\psi(z^n)\\right)
            \\right]

    **Deliberate deviation from the drafted spec** (which used the original
    input ``f(x^n)`` as baseline): using the *reconstruction* ``f(D_ψ(z^n))`` as
    baseline isolates the causal effect of the traversal from the decoder's
    reconstruction error, which would otherwise flip labels even at ``δ=0`` and
    inflate every ICC score toward the reconstruction-flip rate (destroying the
    parent-vs-non-parent contrast). Documented in
    ``docs/spec_code_reconciliation.md`` §2.

    Under identifiability conditions (Hyvärinen 2019; Song 2024) a high
    :math:`\\mathrm{ICC}_i` means concept *i* is causally relevant to the
    classifier's decision up to permutation and element-wise reparameterisation.
    **Latent dimensions are identified only up to permutation**, so a caller
    comparing "causal-parent vs non-parent" dims MUST first align latents to
    ground-truth factors (e.g. via :func:`mcc` + Hungarian) — ``icc_scores[i]``
    is *not* factor *i*.

    Parameters
    ----------
    X:
        Input time series, shape ``(N, T, k)``.
    encoder:
        Callable ``(N, T, k) → (N, d_z)`` — posterior-mean latent codes.
    decoder:
        Callable ``(N, d_z) → (N, T, k)`` — **deterministic** decode (no
        reparam sampling), else ICC is noisy.
    classifier:
        Black-box ``f`` exposing ``predict`` (or directly callable), accepting
        ``(N, T, k)`` and returning ``(N,)`` integer labels.
    delta:
        Traversal magnitude. If ``scale_by_std`` (default), this is a unitless
        factor ``c`` and the per-dimension step is ``δ_i = c · std(Z[:, i])``
        (so a fixed raw ``δ`` does not confound causal relevance with each
        latent's heterogeneous scale). If ``scale_by_std=False``, ``delta`` is
        the raw step (scalar or ``(d_z,)`` array).
    scale_by_std:
        Scale the step per dimension by the dataset latent std (recommended).
    symmetric:
        Average the flip rate over ``+δ`` and ``−δ`` (recommended — a signed
        concept can flip in only one direction).

    Returns
    -------
    icc_scores : ndarray of shape ``(d_z,)`` in ``[0, 1]``. Higher ⇒ the concept
        is more causally relevant to the classifier.
    """
    X_arr = np.asarray(X, dtype=float)
    predict = getattr(classifier, "predict", classifier)

    Z = np.asarray(encoder(X_arr), dtype=float)  # (N, d_z)
    d_z = Z.shape[1]

    if scale_by_std:
        std = Z.std(axis=0)
        std = np.where(std > 0, std, 1.0)  # guard dead dims
        step = np.asarray(delta, dtype=float) * std
    else:
        step = np.broadcast_to(np.asarray(delta, dtype=float), (d_z,)).astype(float)

    # Correction 1: matched baseline — prediction on the *unperturbed
    # reconstruction*, so ICC measures only the traversal, not reconstruction error.
    f_baseline = np.asarray(predict(np.asarray(decoder(Z), dtype=float))).reshape(-1)

    signs = (1.0, -1.0) if symmetric else (1.0,)
    icc_scores = np.zeros(d_z)
    for i in range(d_z):
        flips = 0.0
        for s in signs:
            Z_pert = Z.copy()
            Z_pert[:, i] += s * step[i]
            f_pert = np.asarray(predict(np.asarray(decoder(Z_pert), dtype=float))).reshape(-1)
            flips += float(np.mean(f_pert != f_baseline))
        icc_scores[i] = flips / len(signs)

    return icc_scores


# ---------------------------------------------------------------------------
# MIG — Mutual Information Gap
# ---------------------------------------------------------------------------


def mig(
    z_inferred: np.ndarray,
    z_true: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Mutual Information Gap (MIG).

    For each ground-truth generative factor *k* the MIG measures the
    normalised gap between the highest and second-highest mutual information
    with any inferred latent dimension:

    .. math::

        \\mathrm{MIG} = \\frac{1}{K} \\sum_k \\frac{1}{H(v_k)}
            \\left(
                I(z_{j^*_k}; v_k) - \\max_{j \\neq j^*_k} I(z_j; v_k)
            \\right)

    where :math:`j^*_k = \\arg\\max_j I(z_j; v_k)`.

    A score of 1 means every factor is captured by a unique latent; 0 means
    information is spread uniformly across dimensions.

    Parameters
    ----------
    z_inferred:
        Inferred latent codes, shape ``(N, d_z)``.
    z_true:
        Ground-truth generative factors, shape ``(N, K)``.
    n_bins:
        Number of histogram bins used for MI estimation.

    Returns
    -------
    float
        MIG score in ``[0, 1]``.
    """
    Z = np.asarray(z_inferred, dtype=float)
    V = np.asarray(z_true, dtype=float)
    _, d_z = Z.shape
    K = V.shape[1]

    # MI matrix: (K, d_z)
    MI = np.zeros((K, d_z))
    for k in range(K):
        for j in range(d_z):
            MI[k, j] = _mi_binned(Z[:, j], V[:, k], n_bins=n_bins)

    # Entropy of each true factor
    H = np.array([_marginal_entropy_binned(V[:, k], n_bins=n_bins) for k in range(K)])

    gaps = np.zeros(K)
    for k in range(K):
        sorted_mi = np.sort(MI[k])[::-1]
        top1 = sorted_mi[0]
        top2 = sorted_mi[1] if d_z >= 2 else 0.0
        gaps[k] = (top1 - top2) / (H[k] + 1e-12)

    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# DCI — Disentanglement and Completeness
# ---------------------------------------------------------------------------


def dci(
    z_inferred: np.ndarray,
    z_true: np.ndarray,
    regressor_kwargs: Optional[dict] = None,
    random_state: int = 0,
) -> dict[str, float]:
    """Disentanglement (DCI-D) and Completeness (DCI-C).

    Fits a :class:`~sklearn.ensemble.GradientBoostingRegressor` to predict
    each true factor from inferred latents and builds the importance matrix
    ``R`` where ``R[k, j]`` is the importance of latent *j* for predicting
    factor *k*.

    * **Disentanglement** ``D_j = 1 - H(\\hat{P}_{\\cdot j})`` where
      :math:`\\hat{P}_{\\cdot j}` is the column-normalised weight vector.
      A latent with all weight on a single factor has D = 1 (perfectly
      disentangled); uniform weight gives D = 0.

    * **Completeness** ``C_k = 1 - H(\\hat{P}_{k \\cdot})`` where
      :math:`\\hat{P}_{k \\cdot}` is the row-normalised weight vector.
      A factor captured by exactly one latent has C = 1; captured by all
      latents equally gives C = 0.

    Parameters
    ----------
    z_inferred:
        Inferred latent codes, shape ``(N, d_z)``.
    z_true:
        Ground-truth generative factors, shape ``(N, K)``.
    regressor_kwargs:
        Extra kwargs forwarded to :class:`GradientBoostingRegressor`.  The
        ``random_state`` parameter is set separately.
    random_state:
        Random seed for the regressors.

    Returns
    -------
    dict with keys:
        ``"disentanglement"`` — mean D score across latents, in ``[0, 1]``.
        ``"completeness"``    — mean C score across factors, in ``[0, 1]``.
        ``"R"``               — raw importance matrix of shape ``(K, d_z)``.
    """
    Z = np.asarray(z_inferred, dtype=float)
    V = np.asarray(z_true, dtype=float)
    K = V.shape[1]
    d_z = Z.shape[1]

    kwargs = dict(n_estimators=100, max_depth=3)
    if regressor_kwargs:
        kwargs.update(regressor_kwargs)
    kwargs["random_state"] = random_state

    # Build importance matrix R: (K, d_z)
    R = np.zeros((K, d_z))
    for k in range(K):
        reg = GradientBoostingRegressor(**kwargs)
        reg.fit(Z, V[:, k])
        R[k] = reg.feature_importances_

    # Disentanglement: entropy of column-normalised R, per latent j
    R_col = _normalise_cols(R)
    log2_dz = np.log(d_z) if d_z > 1 else 1.0
    D = np.array([
        1.0 - _entropy(R_col[:, j]) / log2_dz
        for j in range(d_z)
    ])

    # Completeness: entropy of row-normalised R, per factor k
    R_row = _normalise_rows(R)
    log2_K = np.log(K) if K > 1 else 1.0
    C = np.array([
        1.0 - _entropy(R_row[k]) / log2_K
        for k in range(K)
    ])

    return {
        "disentanglement": float(np.mean(D)),
        "completeness": float(np.mean(C)),
        "R": R,
    }


# ---------------------------------------------------------------------------
# MCC — Mean Correlation Coefficient
# ---------------------------------------------------------------------------


def mcc(
    z_inferred: np.ndarray,
    z_true: np.ndarray,
) -> float:
    """Mean Correlation Coefficient (MCC).

    Computes the absolute Pearson correlation matrix between every pair of
    inferred and true latent dimensions, then solves the optimal linear
    assignment (Hungarian algorithm) to maximally match dimensions.  The
    mean absolute correlation of matched pairs is returned.

    This is the standard identifiability metric used in nonlinear-ICA
    literature (Hyvärinen 2019; Khemakhem et al. 2020) to verify that
    inferred latents recover true factors up to permutation and monotone
    reparameterisation.

    Parameters
    ----------
    z_inferred:
        Inferred latent codes, shape ``(N, d_z)``.
    z_true:
        Ground-truth generative factors, shape ``(N, K)``.
        If ``d_z != K`` the smaller dimension is used (unmatched columns are
        ignored).

    Returns
    -------
    float
        Mean absolute Pearson correlation of optimally matched pairs, in
        ``[0, 1]``.  1 means perfect recovery.
    """
    Z = np.asarray(z_inferred, dtype=float)
    V = np.asarray(z_true, dtype=float)
    d_z = Z.shape[1]
    K = V.shape[1]

    # Absolute correlation matrix C[j, k] = |corr(Z[:, j], V[:, k])|
    corr_mat = np.zeros((d_z, K))
    for j in range(d_z):
        for k in range(K):
            r, _ = pearsonr(Z[:, j], V[:, k])
            corr_mat[j, k] = abs(r) if np.isfinite(r) else 0.0

    # Optimal assignment: maximise sum of correlations = minimise negative
    row_ind, col_ind = linear_sum_assignment(-corr_mat)
    matched_corrs = corr_mat[row_ind, col_ind]
    return float(matched_corrs.mean())


# ---------------------------------------------------------------------------
# Bench-ported additions
# ---------------------------------------------------------------------------


def icc(attribution: np.ndarray, int_channel: int) -> float:
    """Intervention-Channel Consistency (attribution-mass variant).

    Fraction of total attribution mass that falls on the ground-truth
    intervened channel. Ported from causal_tscf_bench/metrics/axis_a.py.

    Parameters
    ----------
    attribution : (T, k)
    int_channel : ground-truth intervened channel index

    Returns
    -------
    float in [0, 1]
    """
    total_mass = np.abs(attribution).sum()
    if total_mass < 1e-12:
        return 0.0
    channel_mass = np.abs(attribution[:, int_channel]).sum()
    return float(channel_mass / total_mass)


def mcc_concept(
    attribution: np.ndarray,
    causal_parents: list,
) -> float:
    """Causal Coverage — chance-normalized attribution mass on causal parents.

    .. math::

        \\mathrm{MCC_{cov}} = \\frac{\\sum_{p \\in \\mathrm{Pa}} \\sum_t
        |a_{t,p}| \\; / \\; \\sum_{j,t} |a_{t,j}|}{|\\mathrm{Pa}| / k}

    i.e. the share of total attribution mass landing on ground-truth
    causal-parent channels, divided by the share a *uniform* map would place
    there. ``1.0`` = chance-level alignment; ``> 1`` = attribution
    concentrates on causal parents; ``< 1`` = attribution actively avoids
    them. Maximum is ``k / |Pa|`` (all mass on parents).

    M1 redesign (2026-07-07). The previous formulation — fraction of parents
    whose absolute channel mass exceeded a fixed ``threshold=1e-3`` — had two
    defects: (i) the score depended on the attribution method's output
    *scale*, not its alignment, and (ii) dense saliency maps (IG) cleared any
    reasonable threshold on every channel, pinning the metric at the 1.0
    ceiling with zero discriminative power. A scale-*relative* threshold
    fixes (i) but not (ii) — dense maps still cover everything. The
    continuous mass formulation is threshold-free, scale-invariant by
    construction, mirrors the ``icc`` attribution-mass formulation already
    used on this axis, and its chance normalization makes instances with
    different parent counts comparable under a mean.

    Parameters
    ----------
    attribution    : (T, k)
    causal_parents : list of channel indices that are causal parents (ground-truth)

    Returns
    -------
    float in [0, k/len(causal_parents)]; 1.0 = chance. Returns nan if
    ``causal_parents`` is empty (coverage of zero parents is undefined) or if
    the attribution map carries (numerically) zero total mass.
    """
    if not causal_parents:
        return float("nan")
    attribution = np.asarray(attribution, dtype=float)
    k = attribution.shape[1]
    total_mass = np.abs(attribution).sum()
    if total_mass < 1e-12:
        return float("nan")
    channel_totals = np.abs(attribution).sum(axis=0)  # (k,)
    parent_share = float(sum(channel_totals[p] for p in causal_parents)) / float(total_mass)
    chance_share = len(causal_parents) / k
    return float(parent_share / chance_share)


def latent_disentanglement(Z: np.ndarray, X_channels: np.ndarray) -> float:
    """Latent Disentanglement (LD) via linear R².

    For each causal channel m, fit a linear regression from the best-aligned
    latent dimension to X_channels[:, m] and record R². LD = mean R² over k.

    Parameters
    ----------
    Z          : (N, latent_dim)
    X_channels : (N, k) — ground-truth channel values (e.g. time-mean per channel)

    Returns
    -------
    float in [0, 1]
    """
    from sklearn.linear_model import LinearRegression

    latent_dim = Z.shape[1]
    k = X_channels.shape[1]
    r2_per_channel = []

    for m in range(k):
        y = X_channels[:, m]
        best_r2 = -np.inf
        for d in range(latent_dim):
            reg = LinearRegression().fit(Z[:, [d]], y)
            ss_res = np.sum((y - reg.predict(Z[:, [d]])) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            best_r2 = max(best_r2, r2)
        r2_per_channel.append(max(0.0, best_r2))

    return float(np.mean(r2_per_channel))


def compute_axis_a(
    attributions: np.ndarray,
    int_channels: np.ndarray,
    causal_parents_list: list,
    Z: np.ndarray = None,
    X_channels: np.ndarray = None,
    Z_true: np.ndarray = None,
) -> dict:
    """Aggregate Axis A metrics over N instances.

    Parameters
    ----------
    attributions        : (N, T, k)
    int_channels        : (N,) ground-truth intervened channel per instance
    causal_parents_list : list of length N, each a list of causal parent indices
    Z                   : (N, latent_dim) optional; encoder outputs for LD/MCC
    X_channels          : (N, k) optional; ground-truth channel means for LD
    Z_true              : (N, K) optional; ground-truth latent factors for MCC

    Returns
    -------
    dict with keys: ICC, MCC_coverage, LD, MCC_disent
    """
    N = attributions.shape[0]
    icc_vals, mcc_vals = [], []

    for i in range(N):
        icc_vals.append(icc(attributions[i], int(int_channels[i])))
        mc = mcc_concept(attributions[i], causal_parents_list[i])
        if not np.isnan(mc):
            mcc_vals.append(mc)

    ld = float("nan")
    if Z is not None and X_channels is not None:
        ld = latent_disentanglement(Z, X_channels)

    mcc_disent = float("nan")
    if Z_true is not None and Z is not None and Z_true.shape == Z.shape:
        mcc_disent = mcc(Z, Z_true)

    return {
        "ICC": float(np.mean(icc_vals)),
        "MCC_coverage": float(np.mean(mcc_vals)) if mcc_vals else float("nan"),
        "LD": ld,
        "MCC_disent": mcc_disent,
    }
