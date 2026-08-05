"""Bootstrap confidence intervals for aggregated benchmark metrics (M2, O2).

Every headline number reported from M2 onward carries a 95% CI (``docs/05_evaluation_plan.md`` §6,
Statistical Requirements). Two estimators are provided:

* :func:`bootstrap_ci` — a plain percentile bootstrap (Efron, 1979) over a
  flat, i.i.d. sample of per-instance values. Appropriate for a *single*
  seed's per-instance metric values, or any genuinely exchangeable sample.
* :func:`hierarchical_bootstrap_ci` — a two-level *cluster* bootstrap over
  seed-grouped per-instance values: resample the seeds with replacement, then
  resample the instances *within* each chosen seed's run with replacement,
  pool, and recompute the statistic. This is the estimator the multi-seed
  protocol needs: instances within one seed's run share that seed's SCM draw,
  classifier fit, and CF-selection order, so they are not independent of each
  other across seeds. A flat bootstrap that pools every instance from every
  seed into one i.i.d. sample would silently understate the reported
  uncertainty whenever seeds differ systematically (e.g. one seed's LSTM fit
  happens to be worse, or one seed's SCM draw happens to be easier for a
  given CF method) — exactly the between-run variability the multi-seed
  protocol exists to quantify.

Design choice (one-line rationale): a hand-rolled percentile bootstrap is used
instead of ``scipy.stats.bootstrap`` because the latter has no first-class
support for *two-level* cluster resampling (resample groups, then resample
within the resampled groups) — the hierarchical structure here (seed, then
instance) needs that explicitly; ``scipy.stats.bootstrap`` is built for a
flat i.i.d. sample (or matched paired samples), which is exactly the case
:func:`bootstrap_ci` covers, but not the multi-seed case.

Both estimators are deterministic given the same ``seed`` argument (a
:class:`numpy.random.Generator` seed for the *resampling*, unrelated to any
experiment seed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate + percentile bootstrap CI for one metric."""

    mean: float
    ci_lo: float
    ci_hi: float
    n: int  #: total number of underlying (non-NaN) observations
    n_boot: int  #: number of bootstrap resamples used

    def as_dict(self, prefix: str = "") -> dict:
        """Flat ``{prefix + 'mean': ..., prefix + 'ci_lo': ..., ...}`` dict —
        convenient for widening into a CSV row (one row per method)."""
        return {
            f"{prefix}mean": self.mean,
            f"{prefix}ci_lo": self.ci_lo,
            f"{prefix}ci_hi": self.ci_hi,
            f"{prefix}n": self.n,
        }


_NAN_RESULT_TEMPLATE = dict(mean=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), n=0)


def _percentile_ci(boot_stats: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = 1.0 - ci
    lo = float(np.nanpercentile(boot_stats, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> BootstrapResult:
    """Percentile bootstrap CI for a flat, i.i.d. sample of per-instance values.

    Parameters
    ----------
    values:
        1-D sequence of per-instance metric values (e.g. one method's
        per-instance ``validity`` column for a single seed's run). ``NaN``
        entries are dropped before resampling (matches ``aggregate_method_row``'s
        existing convention of skipping ``None``/``""`` entries).
    n_boot:
        Number of bootstrap resamples.
    ci:
        Confidence level (default 0.95 → a 95% CI, the plan's standing
        requirement from M2 onward).
    seed:
        Seed for the resampling RNG (deterministic given the same ``values``
        and ``seed`` — unrelated to any experiment/SCM seed).
    statistic:
        Callable reducing a 1-D array to a scalar (default ``np.mean``, the
        quantity every aggregated table column already reports).

    Returns
    -------
    BootstrapResult
        ``n=0`` (all-NaN result) if every value is missing — never fabricated.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return BootstrapResult(n_boot=n_boot, **_NAN_RESULT_TEMPLATE)
    point = float(statistic(arr))
    if n == 1:
        # A single observation has no resampling variability to estimate;
        # report a degenerate point CI rather than fabricate spread.
        return BootstrapResult(mean=point, ci_lo=point, ci_hi=point, n=1, n_boot=n_boot)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(arr[idx[b]]) for b in range(n_boot)], dtype=float)
    lo, hi = _percentile_ci(boot, ci)
    return BootstrapResult(mean=point, ci_lo=lo, ci_hi=hi, n=n, n_boot=n_boot)


def hierarchical_bootstrap_ci(
    groups: Sequence[Sequence[float]],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> BootstrapResult:
    """Two-level (seed-cluster) bootstrap CI over seed-grouped per-instance values.

    Parameters
    ----------
    groups:
        A sequence of 1-D arrays, one per seed — each the per-instance metric
        values for that seed's run (may differ in length across seeds, e.g. if
        a CF method failed on some instances in one seed's run). ``NaN``
        entries are dropped within each group before resampling; empty groups
        (all-NaN or zero-length) are dropped entirely.
    n_boot, ci, seed, statistic:
        See :func:`bootstrap_ci`.

    Returns
    -------
    BootstrapResult
        ``n`` is the *total* pooled observation count across all seeds
        (informational — the resampling unit is the seed, not the instance).
        ``n=0`` if no seed contributed any non-NaN observation.

    Protocol
    --------
    For each of ``n_boot`` iterations: (1) resample the seeds with
    replacement (same count as the number of input groups); (2) for each
    chosen seed, resample its own instances with replacement (same count as
    that seed's group size); (3) pool every resampled instance across the
    resampled seeds and recompute ``statistic`` once on the pooled sample.
    This is the standard hierarchical/cluster bootstrap for two-level
    (seed, instance) data (see module docstring for why a flat pooled
    bootstrap is not appropriate here).
    """
    clean_groups = [np.asarray(g, dtype=float) for g in groups]
    clean_groups = [g[~np.isnan(g)] for g in clean_groups]
    clean_groups = [g for g in clean_groups if g.size > 0]
    n_groups = len(clean_groups)
    n_total = int(sum(g.size for g in clean_groups))

    if n_groups == 0 or n_total == 0:
        return BootstrapResult(n_boot=n_boot, **_NAN_RESULT_TEMPLATE)

    pooled_point = np.concatenate(clean_groups)
    point = float(statistic(pooled_point))

    if n_groups == 1 and clean_groups[0].size == 1:
        # A single seed with a single observation: no resampling variability
        # to estimate at either level; report a degenerate point CI.
        return BootstrapResult(mean=point, ci_lo=point, ci_hi=point, n=1, n_boot=n_boot)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        pooled = []
        for gi in chosen:
            g = clean_groups[gi]
            pooled.append(g[rng.integers(0, g.size, size=g.size)])
        boot[b] = statistic(np.concatenate(pooled))

    lo, hi = _percentile_ci(boot, ci)
    return BootstrapResult(mean=point, ci_lo=lo, ci_hi=hi, n=n_total, n_boot=n_boot)


@dataclass(frozen=True)
class CrossingResult:
    """Point estimate + bootstrap CI for where a curve crosses a threshold."""

    horizon: float  #: NaN if the pooled curve never crosses within the tested range
    ci_lo: float
    ci_hi: float
    n_seeds: int
    n_boot: int
    frac_boot_crossed: float  #: fraction of bootstrap resamples that found a crossing

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}horizon": self.horizon,
            f"{prefix}ci_lo": self.ci_lo,
            f"{prefix}ci_hi": self.ci_hi,
            f"{prefix}n_seeds": self.n_seeds,
            f"{prefix}frac_boot_crossed": self.frac_boot_crossed,
        }


def _first_falling_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Smallest ``x`` at which ``y`` first drops to/below ``threshold``, linearly
    interpolated between the bracketing grid points. ``x`` must be sorted
    ascending. Returns ``nan`` if ``y`` never crosses (stays above threshold at
    every point, or is already at/below it at the first point -- both are
    genuinely unobserved crossings, not a crossing at ``x[0]``: the curve may
    have crossed *before* the first tested horizon, which this grid cannot
    see, so reporting ``x[0]`` would fabricate precision the data doesn't
    support).
    """
    if y[0] <= threshold:
        return float("nan")
    for i in range(1, len(x)):
        if y[i] <= threshold:
            # Linear interpolation between (x[i-1], y[i-1]) and (x[i], y[i]).
            if y[i - 1] == y[i]:
                return float(x[i])
            frac = (y[i - 1] - threshold) / (y[i - 1] - y[i])
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))
    return float("nan")


def collapse_horizon_ci(
    horizons: Sequence[float],
    validity_by_seed: Sequence[Sequence[float]],
    threshold: float = 0.5,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> CrossingResult:
    """Bootstrap CI for the horizon at which a validity curve collapses.

    Unlike :func:`bootstrap_ci` / :func:`hierarchical_bootstrap_ci`, the
    quantity being estimated here is a property of the **seed-pooled mean
    curve** (where does it cross ``threshold``), not a mean of per-instance
    values -- so the resampling unit is the seed alone; there is no
    within-seed instance level to resample a second time, because a "collapse
    horizon" is not defined per instance.

    Parameters
    ----------
    horizons:
        Sorted ascending ``T - t0`` grid points swept (e.g. from
        ``table_horizon_*.csv``'s ``T_minus_t0`` column).
    validity_by_seed:
        One row per seed, each a sequence of per-horizon validity means
        aligned to ``horizons`` (same length, same order). A seed missing a
        horizon should not be passed at all for that row's construction —
        this function assumes a complete grid per seed.
    threshold:
        Validity level defining "collapsed" (default 0.5).
    n_boot, ci, seed:
        As in :func:`bootstrap_ci`.

    Returns
    -------
    CrossingResult
        ``horizon`` is the pooled-curve crossing point (mean validity across
        all seeds at each horizon, then interpolated crossing). ``ci_lo``/
        ``ci_hi`` come from resampling seeds with replacement and
        recomputing the crossing on the resampled pooled curve each time;
        resamples that never cross contribute ``nan`` and are excluded by
        ``nanpercentile`` rather than treated as a crossing at either
        boundary. ``frac_boot_crossed`` reports how many resamples actually
        found a crossing -- a CI built from a small fraction is a much
        weaker claim than one built from nearly all of them, so it ships
        alongside the interval rather than being silently absorbed into it.
    """
    x = np.asarray(horizons, dtype=float)
    order = np.argsort(x)
    x = x[order]
    Y = np.asarray(validity_by_seed, dtype=float)[:, order]  # (n_seeds, n_horizons)
    n_seeds = Y.shape[0]

    if n_seeds == 0 or Y.shape[1] != len(x):
        return CrossingResult(
            horizon=float("nan"),
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            n_seeds=n_seeds,
            n_boot=n_boot,
            frac_boot_crossed=0.0,
        )

    point = _first_falling_crossing(x, Y.mean(axis=0), threshold)

    if n_seeds == 1:
        crossed = not np.isnan(point)
        return CrossingResult(
            horizon=point,
            ci_lo=point,
            ci_hi=point,
            n_seeds=1,
            n_boot=n_boot,
            frac_boot_crossed=1.0 if crossed else 0.0,
        )

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, n_seeds, size=n_seeds)
        boot[b] = _first_falling_crossing(x, Y[chosen].mean(axis=0), threshold)

    frac_crossed = float(np.mean(~np.isnan(boot)))
    if frac_crossed == 0.0:
        lo = hi = float("nan")
    else:
        lo, hi = _percentile_ci(boot, ci)
    return CrossingResult(
        horizon=point,
        ci_lo=lo,
        ci_hi=hi,
        n_seeds=n_seeds,
        n_boot=n_boot,
        frac_boot_crossed=frac_crossed,
    )
