"""
Axis A — Concept Quality Metrics.

Metrics:
  ICC (attribution): Fraction of attribution mass on the intervened channel.
  ICC (latent):      Decoder-intervention version — ICC = (1/N) Σ_n 1[f(D(z+δe_i)) ≠ f(x)].
  MCC (causal):      Fraction of causal parent channels with non-zero attribution mass.
  MCC (disentangle): Hyvärinen's Mean Correlation Coefficient — optimal assignment over
                     the cross-correlation matrix of true vs. predicted latent factors.
  LD:                Linear R² alignment between latent dims and ground-truth channels.

Reference:
  Lippe et al. (2022) CITRIS.
  Hyvärinen & Morioka (2016): MCC for disentanglement evaluation.
  Montavon et al. (2018) Explanation Methods review.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def icc(attribution: np.ndarray, int_channel: int) -> float:
    """
    Intervention-Channel Consistency (attribution mass variant).

    Measures what fraction of the explanation's total attribution mass falls
    on the ground-truth intervened channel.

    Parameters
    ----------
    attribution : (T, M)  attribution map
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


def icc_latent(encoder, decoder, classifier, X: np.ndarray,
               int_dim: int, delta: float = 2.0) -> float:
    """
    Intervention-Channel Consistency (latent decoder-space variant).

    ICC = (1/N) Σ_n 1[ f(D_ψ(z^n + δ·e_i)) ≠ f(x^n) ]

    Measures whether intervening on latent dimension `int_dim` (which should
    correspond to the causal concept) causes a prediction flip.

    Parameters
    ----------
    encoder    : callable (T, M) -> (latent_dim,)  numpy arrays
    decoder    : callable (latent_dim,) -> (T, M)  numpy arrays
    classifier : TSClassifier with .predict()
    X          : (N, T, M) test instances
    int_dim    : latent dimension index to intervene on
    delta      : intervention magnitude (additive shift)

    Returns
    -------
    float in [0, 1]; higher = latent dim correctly separates causal concept
    """
    N = X.shape[0]
    count = 0
    for i in range(N):
        x = X[i]                           # (T, M)
        z = encoder(x)                     # (latent_dim,)
        z_int = z.copy()
        z_int[int_dim] += delta
        x_cf = decoder(z_int)              # (T, M)
        pred_orig = classifier.predict(x[np.newaxis])[0]
        pred_cf = classifier.predict(x_cf[np.newaxis])[0]
        if pred_cf != pred_orig:
            count += 1
    return float(count / N)


def mcc_concept(attribution: np.ndarray, causal_parents,
                threshold: float = 1e-3) -> float:
    """
    Causal Coverage MCC — fraction of causal parent channels covered by attribution.

    Parameters
    ----------
    attribution    : (T, M)
    causal_parents : list of channel indices that are causal parents (ground-truth)
    threshold      : minimum absolute attribution to count a channel as "covered"

    Returns
    -------
    float in [0, 1]; returns nan if causal_parents is empty
    """
    if not causal_parents:
        return float("nan")
    channel_totals = np.abs(attribution).sum(axis=0)  # (M,)
    covered = sum(1 for p in causal_parents if channel_totals[p] > threshold)
    return float(covered / len(causal_parents))


def mcc_disentanglement(Z_true: np.ndarray, Z_pred: np.ndarray) -> float:
    """
    Mean Correlation Coefficient for latent disentanglement (Hyvärinen's MCC).

    Finds the optimal one-to-one alignment between true and predicted latent
    factors via the Hungarian algorithm on the absolute correlation matrix.

    MCC = (1/K) max_{σ} Σ_i |corr(z_true_i, z_pred_{σ(i)})|

    Parameters
    ----------
    Z_true : (N, K) ground-truth latent factors
    Z_pred : (N, K) predicted latent factors (e.g. from iVAE encoder)

    Returns
    -------
    float in [0, 1]; 1 = perfect disentanglement
    """
    from scipy.optimize import linear_sum_assignment

    K = Z_true.shape[1]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            r = np.corrcoef(Z_true[:, i], Z_pred[:, j])[0, 1]
            C[i, j] = abs(r) if np.isfinite(r) else 0.0

    row_ind, col_ind = linear_sum_assignment(-C)   # maximise
    return float(C[row_ind, col_ind].mean())


def latent_disentanglement(Z: np.ndarray, X_channels: np.ndarray) -> float:
    """
    Latent Disentanglement (LD) via linear R².

    For each causal channel m, fit a linear regression from the best-aligned
    latent dimension to X_channels[:, m] and record R². LD = mean R² over M.

    Parameters
    ----------
    Z          : (N, latent_dim)
    X_channels : (N, M) — ground-truth channel values (e.g. time-mean per channel)

    Returns
    -------
    float in [0, 1]
    """
    N, latent_dim = Z.shape
    M = X_channels.shape[1]
    r2_per_channel = []

    for m in range(M):
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


def compute_axis_a(attributions: np.ndarray, int_channels: np.ndarray,
                   causal_parents_list,
                   Z: np.ndarray = None,
                   X_channels: np.ndarray = None,
                   Z_true: np.ndarray = None) -> dict:
    """
    Aggregate Axis A over N instances.

    Parameters
    ----------
    attributions        : (N, T, M)
    int_channels        : (N,) ground-truth intervened channel per instance
    causal_parents_list : list of length N, each a list of causal parent indices
    Z                   : (N, latent_dim) optional; encoder outputs for LD
    X_channels          : (N, M) optional; ground-truth channel means for LD
    Z_true              : (N, K) optional; ground-truth latent factors for MCC_disent

    Returns
    -------
    dict with keys: 'ICC', 'MCC_coverage', 'LD', 'MCC_disent'
      - ICC           : attribution mass on intervened channel (mean over N)
      - MCC_coverage  : fraction of causal parents covered by attribution
      - LD            : linear R² latent alignment (nan if Z not provided)
      - MCC_disent    : Hyvärinen's MCC between Z_true and Z (nan if not provided)
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
        mcc_disent = mcc_disentanglement(Z_true, Z)

    return {
        "ICC": float(np.mean(icc_vals)),
        "MCC_coverage": float(np.mean(mcc_vals)) if mcc_vals else float("nan"),
        "LD": ld,
        "MCC_disent": mcc_disent,
    }
