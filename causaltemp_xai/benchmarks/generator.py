"""Structural-causal-model generators for temporal data.

* :class:`LinearSCMT` — VAR(L) process; each variable ``x_i`` at time ``t`` is a
  linear function of its causal parents at lags 1…L plus non-Gaussian noise::

      x_t  =  sum_{l=1}^{L}  A_l @ x_{t-l}  +  eps_t

* :class:`NlinearSCMT` — same graph machinery and noise, but the linear
  mechanism is swapped for an additive-noise per-node MLP
  (:class:`~causaltemp_xai.benchmark.mechanisms.MLPMechanism`)::

      x_t  =  mechanism.forward_numpy(window_{t-L..t-1})  +  eps_t

* :class:`RegimeSwitchNlinearSCMT` — M4/H7 ablation: two-regime structural
  break layered on :class:`NlinearSCMT`. Same causal graph throughout; only
  the per-node MLP mechanism's numeric parameters (decay/gain/weights)
  switch, once, at a fixed deterministic timestep partway through each
  trajectory's *observed* horizon. Minimal viable design (two regimes, one
  switch point — no HMM, no learned transition probabilities); see
  ``docs/m4_ablation_presets_smoke.md``.

* :class:`HMMRegimeSwitchNlinearSCMT` — M4/H7 ablation, HMM variant: the same
  parameter-only regime switch, but driven by a genuine **hidden Markov
  process** (``R in {2, 3}`` regimes, an ``R x R`` transition matrix, random
  per-sequence change-points) instead of a single deterministic break. Kept
  alongside :class:`RegimeSwitchNlinearSCMT` (not replacing it) so the
  deterministic-single-break vs. stochastic-multi-break contrast is a clean
  A/B for H7.

Both :class:`LinearSCMT` and :class:`NlinearSCMT` share the graph sampler
(:func:`_sample_graph`) so the no-self-loop-at-lag-1 rule and the per-lag
Bernoulli draw are identical, and ``eps_t`` is drawn from one of three
innovation distributions: Laplace (default), Uniform, or Gaussian
(``noise_type="gaussian"`` — M4/H5 negative-control ablation, added
2026-07-08; variance-matched to the default Laplace scale so the ablation
isolates innovation *shape*, not scale — see :data:`_GAUSSIAN_STD` and
``docs/m4_ablation_presets_smoke.md``).

Scope (this iteration)
----------------------
:class:`NlinearSCMT` ships nonlinear **transitions** with **additive noise**
only. Additive noise makes Pearl abduction an exact subtraction
(``eps = x − f(parents)``), so CF-faith and the oracle structural-CF
(:mod:`causaltemp_xai.benchmark.structural_cf`) remain valid on the nonlinear
mechanisms. Two axes are deliberately **out of scope** and left as future
stress tests (see ``docs/plans/nlinearscm-t/`` Backlog #1): nonlinear *mixing*
``x = g(z)`` (an invertible observation map over latents — iVAE/CITRIS
identifiability) and *non-additive* (location-scale) noise. Both degrade
abduction from exact to **partial** identification, which is why they are not
mixed into this additive-transition benchmark.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from causaltemp_xai.benchmarks.mechanisms import LinearMechanism, MLPMechanism

_NOISE_TYPES = ("laplace", "uniform", "gaussian")

#: Std-dev for the Gaussian innovation branch (M4/H5 negative-control
#: ablation). Chosen to match the *variance* of the default
#: ``laplace(loc=0, scale=0.1)`` innovation exactly — a Laplace(0, b)'s
#: variance is ``2 * b**2``, so ``std = b * sqrt(2)`` — so the ablation
#: isolates innovation-distribution *shape* (kurtosis), not scale. See
#: ``docs/m4_ablation_presets_smoke.md`` for the derivation and the
#: pre-registered H5 expected direction.
_GAUSSIAN_STD = 0.1 * float(np.sqrt(2.0))


def _sample_lag_mask(
    k: int, lag_index: int, sparsity: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw one lag's Bernoulli edge mask, zeroing the diagonal at lag 1.

    A single ``rng.random((k, k))`` draw — the atomic RNG step that both
    generators share. Keeping the draw factored here (rather than pre-drawing all
    lags) lets :class:`LinearSCMT` interleave it with its coefficient draws and so
    preserve the frozen v0.1 RNG sequence (see the golden test); ``lag_index == 0``
    is the lag-1 slice, where the no-self-loop rule zeros the diagonal.
    """
    mask = (rng.random((k, k)) < sparsity).astype(float)
    if lag_index == 0:
        np.fill_diagonal(mask, 0.0)
    return mask


def _sample_graph(k: int, L: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a lagged adjacency tensor ``(k, k, L)`` from ``rng``.

    Draws one Bernoulli mask per lag via :func:`_sample_lag_mask` (no self-loop at
    lag 1). Used directly by :class:`NlinearSCMT`; :class:`LinearSCMT` shares the
    per-lag :func:`_sample_lag_mask` (interleaved with coefficient draws) instead
    of calling this, to keep its exact RNG draw order frozen.
    """
    graph = np.zeros((k, k, L), dtype=float)
    for lag in range(L):
        graph[:, :, lag] = _sample_lag_mask(k, lag, sparsity, rng)
    return graph


class LinearSCMT:
    """VAR(L) data generator with a fixed causal structure.

    Parameters
    ----------
    k:
        Number of variables (dimensions) in the time series.
    L:
        Maximum lag of the VAR process.
    sparsity:
        Probability that any given lagged edge is present in the causal graph.
        Value in ``(0, 1]``; lower values produce sparser graphs.
    noise_type:
        Noise distribution for the innovation terms. One of ``"laplace"``
        (default), ``"uniform"``, or ``"gaussian"`` — the last is the M4/H5
        negative-control ablation, variance-matched to the Laplace default
        (see :data:`_GAUSSIAN_STD`).
    T:
        Default trajectory length used by :meth:`generate`.
    N:
        Default number of samples used by :meth:`generate`.
    seed:
        Random seed for reproducibility.  Pass ``None`` for non-deterministic
        behaviour.
    """

    def __init__(
        self,
        k: int = 5,
        L: int = 2,
        sparsity: float = 0.3,
        noise_type: Literal["laplace", "uniform", "gaussian"] = "laplace",
        T: int = 50,
        N: int = 200,
        seed: Optional[int] = 42,
    ) -> None:
        if noise_type not in _NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {_NOISE_TYPES}, got {noise_type!r}")
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # Build the lagged graph and mechanism once at construction.
        self.graph, self.mechanism = self._build_graph_and_mechanisms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, burn_in: int = 100) -> dict:
        """Generate a full dataset from the LinearSCM-T model.

        Parameters
        ----------
        burn_in:
            Number of initial time steps discarded to remove transient effects.

        Returns
        -------
        dict with keys:

        ``"X"``
            Float array of shape ``(N, T, k)`` – the observed time series.
        ``"Y"``
            Integer array of shape ``(N,)`` – binary labels derived from a
            threshold on the final-timestep latent value of variable 0.
        ``"graph"``
            Binary adjacency tensor of shape ``(k, k, L)``.  ``graph[i, j, l]``
            is 1 if variable *j* causes variable *i* at lag ``l+1``.
        ``"mechanism"``
            A :class:`~causaltemp_xai.benchmark.mechanisms.Mechanism` (here a
            :class:`LinearMechanism`) producing the deterministic next-step mean.
        """
        total_T = self.T + burn_in
        X_full = np.zeros((self.N, total_T, self.k))

        # Initialise first L steps with small noise
        for lag in range(self.L):
            X_full[:, lag, :] = self._sample_noise(self.N, self.k) * 0.1

        # Simulate VAR(L) forward
        noise = self._sample_noise(self.N * total_T * self.k).reshape(self.N, total_T, self.k)
        for t in range(self.L, total_T):
            # Lag window ordered oldest→newest: (N, L, k). t >= L always holds
            # here, so the window is full (no zero-pad needed).
            window = X_full[:, t - self.L : t, :]
            x_t = noise[:, t, :].copy()
            x_t += self.mechanism.forward_numpy(window)
            X_full[:, t, :] = x_t

        X = X_full[:, burn_in:, :]  # shape (N, T, k)

        # Labels: threshold on final-timestep value of variable 0
        # Use the median across samples so classes are balanced by default
        latent_final = X[:, -1, 0]
        threshold = float(np.median(latent_final))
        Y = (latent_final > threshold).astype(int)

        return {
            "X": X,
            "Y": Y,
            "graph": self.graph,
            "mechanism": self.mechanism,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_graph_and_mechanisms(self) -> tuple[np.ndarray, LinearMechanism]:
        """Sample the causal graph and coefficient matrices.

        Returns
        -------
        graph : ndarray of shape ``(k, k, L)``
        mechanism : LinearMechanism wrapping the L coefficient matrices
        """
        k, L = self.k, self.L
        graph = np.zeros((k, k, L), dtype=float)
        mechanisms = []

        for l in range(L):
            # Bernoulli mask for edges (no self-loops at lag 1). The mask is drawn
            # *before* the coefficients on every lag — that interleaving is the
            # frozen v0.1 RNG sequence the golden test pins, so the draw stays
            # inside this loop (shared atom: `_sample_lag_mask`).
            mask = _sample_lag_mask(k, l, self.sparsity, self._rng)
            graph[:, :, l] = mask

            # Draw coefficients and apply mask
            coefs = self._rng.uniform(-0.5, 0.5, (k, k))
            A = coefs * mask

            # Ensure stationarity: spectral radius < 0.9
            A = _stabilise(A, target_radius=0.9)
            mechanisms.append(A)

        return graph, LinearMechanism(mechanisms)

    def _sample_noise(self, *shape) -> np.ndarray:
        """Draw noise samples from the configured innovation distribution."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        elif self.noise_type == "uniform":
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1
        else:  # gaussian (M4/H5 negative-control ablation)
            return self._rng.normal(loc=0.0, scale=_GAUSSIAN_STD, size=size)


class NlinearSCMT:
    """Nonlinear SCM-T generator: additive-noise per-node MLP transitions.

    Shares :class:`LinearSCMT`'s graph machinery, noise distributions, burn-in and
    median-threshold label rule, but the linear ``A_l @ x`` mechanism is replaced
    by an :class:`~causaltemp_xai.benchmark.mechanisms.MLPMechanism`. The forward
    step is::

        x_t = mechanism.forward_numpy(window_{t-L..t-1}) + eps_t

    The mechanism is contractive (leaky ``decay`` + bounded ``gain·tanh`` output
    branch with spectral-norm-capped weights), so trajectories stay bounded over
    long horizons; a deterministic divergence-resample guard (then a clip
    fallback) catches the rare blow-up without breaking ``same seed ⇒
    identical X``. This boundedness guarantee is unaffected by the ``activation``
    choice (``"tanh"`` or the M4/H6 ``"nonmonotonic"`` ablation) because the
    *output* branch is always ``tanh`` regardless — ``activation`` only selects
    the *hidden*-layer nonlinearity (see
    :class:`~causaltemp_xai.benchmarks.mechanisms.MLPMechanism`).

    Parameters
    ----------
    k, L, sparsity, noise_type, T, N, seed:
        As in :class:`LinearSCMT`.
    hidden:
        Hidden width of each per-node MLP.
    gain:
        Scalar multiplier on the bounded ``tanh`` branch.
    decay_range:
        ``(low, high)`` range for the per-node leaky decay coefficient.
    spectral_cap:
        Spectral-norm cap on each per-node weight matrix (Lipschitz control).
    init_gain:
        Weight-init scale: ``std = init_gain · √(1/fan_in)``.
    activation:
        Hidden-layer activation: ``"tanh"`` (default, monotonic) or
        ``"nonmonotonic"`` (M4/H6 ablation — a bounded ``sin`` activation with
        the same ``|output| <= 1`` range and <=1 Lipschitz bound as ``tanh``,
        so the contractive stability argument is unchanged; only
        monotonicity of the *hidden* representation varies — the mechanism's
        **output** branch is always ``tanh`` regardless of this choice. See
        :mod:`causaltemp_xai.benchmarks.mechanisms` and
        ``docs/m4_ablation_presets_smoke.md``).
    clip:
        Divergence threshold; a trajectory whose ``max|x|`` exceeds ``clip`` (or
        goes non-finite) is resampled, then clipped if still diverging.
    max_resample:
        Maximum divergence-resample rounds before falling back to clipping.
    """

    def __init__(
        self,
        k: int = 5,
        L: int = 2,
        sparsity: float = 0.3,
        noise_type: Literal["laplace", "uniform", "gaussian"] = "laplace",
        T: int = 50,
        N: int = 200,
        seed: Optional[int] = 42,
        hidden: int = 16,
        gain: float = 0.8,
        decay_range: tuple[float, float] = (0.3, 0.8),
        spectral_cap: float = 0.9,
        init_gain: float = 0.7,
        activation: Literal["tanh", "nonmonotonic"] = "tanh",
        clip: float = 1e3,
        max_resample: int = 10,
    ) -> None:
        if noise_type not in _NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {_NOISE_TYPES}, got {noise_type!r}")
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self.hidden = hidden
        self.gain = gain
        self.decay_range = decay_range
        self.spectral_cap = spectral_cap
        self.init_gain = init_gain
        self.activation = activation
        self.clip = clip
        self.max_resample = max_resample
        self._rng = np.random.default_rng(seed)
        # Build graph + mechanism at construction, *before* any noise is drawn, so
        # a noise-only `shifted_config` yields a bit-identical graph + mechanism
        # (Axis-D invariant). Graph first, then MLP weights (both from self._rng).
        self.graph = _sample_graph(self.k, self.L, self.sparsity, self._rng)
        self.mechanism = MLPMechanism.random(
            self.graph,
            hidden=self.hidden,
            rng=self._rng,
            decay_range=self.decay_range,
            gain=self.gain,
            spectral_cap=self.spectral_cap,
            init_gain=self.init_gain,
            activation=self.activation,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, burn_in: int = 100) -> dict:
        """Generate a full nonlinear dataset (same contract as :class:`LinearSCMT`).

        Returns a dict with keys ``"X"`` ``(N, T, k)``, ``"Y"`` ``(N,)``,
        ``"graph"`` ``(k, k, L)`` and ``"mechanism"`` (an ``MLPMechanism``).
        """
        total_T = self.T + burn_in
        X_full = np.zeros((self.N, total_T, self.k))

        # Roll every trajectory, then deterministically resample any that diverge.
        self._roll(X_full, np.arange(self.N), total_T)
        for _ in range(self.max_resample):
            bad = self._diverging(X_full)
            if not bad.any():
                break
            self._roll(X_full, np.nonzero(bad)[0], total_T)
        # Clip fallback: a trajectory still diverging after the resample budget is
        # clamped so the output is always finite and bounded.
        if self._diverging(X_full).any():
            np.clip(X_full, -self.clip, self.clip, out=X_full)

        X = X_full[:, burn_in:, :]  # shape (N, T, k)

        # Labels: median threshold on final-step variable 0 (balanced by default).
        latent_final = X[:, -1, 0]
        threshold = float(np.median(latent_final))
        Y = (latent_final > threshold).astype(int)

        return {
            "X": X,
            "Y": Y,
            "graph": self.graph,
            "mechanism": self.mechanism,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _roll(self, X_full: np.ndarray, rows: np.ndarray, total_T: int) -> None:
        """Simulate the forward dynamics for ``rows`` in place.

        Draws fresh init + innovation noise for exactly these trajectories from the
        seeded generator RNG. ``rows`` is sorted first so the draw order depends
        only on *which* trajectories diverged — never on wall-clock — keeping the
        resample path reproducible under a fixed seed.
        """
        rows = np.sort(np.asarray(rows))
        n = int(rows.size)
        if n == 0:
            return
        sub = np.zeros((n, total_T, self.k))
        # Initialise first L steps with small noise.
        for lag in range(self.L):
            sub[:, lag, :] = self._sample_noise(n, self.k) * 0.1
        noise = self._sample_noise(n * total_T * self.k).reshape(n, total_T, self.k)
        for t in range(self.L, total_T):
            window = sub[:, t - self.L : t, :]  # (n, L, k), oldest→newest
            x_t = noise[:, t, :].copy()
            x_t += self.mechanism.forward_numpy(window)
            sub[:, t, :] = x_t
        X_full[rows] = sub

    def _diverging(self, X_full: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)`` mask of trajectories that blew up or went non-finite."""
        finite = np.isfinite(X_full).all(axis=(1, 2))
        bounded = np.abs(np.nan_to_num(X_full, nan=np.inf)).max(axis=(1, 2)) <= self.clip
        return ~(finite & bounded)

    def _sample_noise(self, *shape) -> np.ndarray:
        """Draw noise from the configured innovation distribution (as linear)."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        elif self.noise_type == "uniform":
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1
        else:  # gaussian (M4/H5 negative-control ablation)
            return self._rng.normal(loc=0.0, scale=_GAUSSIAN_STD, size=size)


#: Default hyperparameters for the M4/H7 regime-switch ablation's two
#: regimes. Regime 1 is deliberately set **identical** to
#: :class:`NlinearSCMT`'s own class defaults (equivalently, ``config.
#: _NL_HYPERPARAMS``) -- so regime 1 *is* the standard, un-ablated nonlinear
#: mechanism family, and (given the same seed and the same graph/``hidden``)
#: draws bit-identical weights to a plain :class:`NlinearSCMT` built with
#: the same seed, since it is the first ``MLPMechanism.random`` call the RNG
#: stream sees in both cases. This isolates the ablation to exactly one
#: thing: the presence of a second, post-switch regime -- not a
#: simultaneous change to the pre-switch dynamics too. Regime 2 then uses a
#: **non-overlapping** ``decay_range`` (with a safety gap, not just a
#: touching boundary) and a distinct ``gain``/``spectral_cap``, so "the two
#: regimes have different effective parameters" is a deterministic, not
#: merely probabilistic, structural property -- see
#: ``tests/test_nlinear_generator.py::TestRegimeSwitchNlinearSCMT`` and
#: ``docs/m4_ablation_presets_smoke.md``. Regime 2 is, if anything, *more*
#: contractive than regime 1 (lower decay ceiling, lower gain, tighter
#: spectral cap) -- a deliberately conservative choice so the ablation
#: cannot itself introduce an instability the stability guard would need to
#: catch.
_REGIME1_DEFAULTS: dict = {
    "decay_range": (0.3, 0.8),
    "gain": 0.8,
    "spectral_cap": 0.9,
    "init_gain": 0.7,
    "activation": "tanh",
}
_REGIME2_DEFAULTS: dict = {
    "decay_range": (0.05, 0.25),
    "gain": 0.35,
    "spectral_cap": 0.5,
    "init_gain": 0.35,
    "activation": "tanh",
}


class RegimeSwitchNlinearSCMT:
    """Two-regime structural-break extension of :class:`NlinearSCMT` (M4/H7).

    Minimal-viable regime-switching ablation: **two** regimes and **one**
    deterministic switch point — no HMM, no learned transition
    probabilities (explicitly out of scope; see
    ``docs/m4_ablation_presets_smoke.md``). Both regimes share the *same*
    causal graph (:func:`_sample_graph`); only the per-node MLP mechanism's
    numeric parameters (``decay``, ``gain``, weights) differ between
    regimes — this is a **parameter** regime switch, not a graph change.

    Regime 1 (:attr:`mechanism1`) governs every timestep ``t < switch_t``
    (including the whole burn-in, so the pre-switch segment starts from a
    settled state); regime 2 (:attr:`mechanism2`) governs ``t >= switch_t``,
    where ``switch_t = burn_in + round(switch_frac * T)`` lands partway
    through the *observed* (post-burn-in) window — by default at its
    midpoint (``switch_frac=0.5``, i.e. "T/2"). This is meant to give
    :func:`causaltemp_xai.eval.shift_vr` (Shift-VR) and CF-faith something
    genuine to detect: a trajectory whose second half was generated by a
    detectably different transition function than a CF method's implicit
    model of the world assumes.

    Downstream single-mechanism contract — read before using CF-faith /
    oracle structural-CF / CARLA-style recourse on this preset
    ---------------------------------------------------------------------
    Every other module in this codebase
    (:mod:`causaltemp_xai.metrics.cf_faith`,
    :mod:`causaltemp_xai.benchmarks.structural_cf`, CARLA's on-manifold
    recourse) assumes **one** time-invariant
    :class:`~causaltemp_xai.benchmarks.mechanisms.Mechanism` per dataset —
    ``Mechanism.forward_numpy`` takes only a lag *window*, with no notion of
    "which regime" or absolute timestep. Rather than invasively add an
    absolute-time argument to that interface everywhere it is called (a
    large, high-blast-radius refactor well beyond a smoke-scale validation
    task), :meth:`generate` returns ``mechanism1`` as the dataset's single
    ``"mechanism"`` — exactly correct for the pre-switch segment, an
    *assumed/nominal* model for the post-switch segment. ``mechanism2`` and
    ``switch_t`` are returned alongside as ground-truth diagnostic metadata,
    not wired into CF-faith/oracle-CF. Concretely: CF-faith/oracle-CF scores
    computed by the standard pipeline on this preset measure faithfulness
    against the **nominal** (regime-1) mechanism, not literally correct
    ground truth for post-switch timesteps — this mismatch is the
    deliberate stress condition H7 is testing for, not a bug, but it means
    CF-faith numbers on this preset are not directly comparable to other
    presets'. A rigorous (non-smoke) H7 study would need a regime-aware
    oracle; that is explicitly out of scope here.

    Parameters
    ----------
    k, L, sparsity, noise_type, T, N, seed:
        As in :class:`NlinearSCMT`.
    hidden:
        Hidden width of each regime's per-node MLP (shared across regimes).
    switch_frac:
        Fraction of the observed window ``T`` after which regime 2 takes
        over. Must be in ``(0, 1)``; default ``0.5`` (the midpoint, per the
        H7 brief's "T/2").
    regime1, regime2:
        Hyperparameter dicts forwarded to
        :meth:`~causaltemp_xai.benchmarks.mechanisms.MLPMechanism.random`
        (``decay_range``, ``gain``, ``spectral_cap``, ``init_gain``,
        ``activation``). Default to :data:`_REGIME1_DEFAULTS` /
        :data:`_REGIME2_DEFAULTS` (non-overlapping ``decay_range`` and
        distinct ``gain``, so the two regimes are deterministically, not
        just statistically, distinguishable).
    clip, max_resample:
        As in :class:`NlinearSCMT`.
    """

    def __init__(
        self,
        k: int = 5,
        L: int = 2,
        sparsity: float = 0.3,
        noise_type: Literal["laplace", "uniform", "gaussian"] = "laplace",
        T: int = 50,
        N: int = 200,
        seed: Optional[int] = 42,
        hidden: int = 16,
        switch_frac: float = 0.5,
        regime1: dict | None = None,
        regime2: dict | None = None,
        clip: float = 1e3,
        max_resample: int = 10,
    ) -> None:
        if noise_type not in _NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {_NOISE_TYPES}, got {noise_type!r}")
        if not (0.0 < switch_frac < 1.0):
            raise ValueError(f"switch_frac must be in (0, 1), got {switch_frac!r}")
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self.hidden = hidden
        self.switch_frac = switch_frac
        self.clip = clip
        self.max_resample = max_resample
        r1 = dict(_REGIME1_DEFAULTS) if regime1 is None else dict(regime1)
        r2 = dict(_REGIME2_DEFAULTS) if regime2 is None else dict(regime2)
        self.regime1_hparams = r1
        self.regime2_hparams = r2
        self._rng = np.random.default_rng(seed)
        # Same graph for both regimes, sampled once before any mechanism
        # weights or noise -- mirrors NlinearSCMT's build order.
        self.graph = _sample_graph(self.k, self.L, self.sparsity, self._rng)
        self.mechanism1 = MLPMechanism.random(self.graph, hidden=self.hidden, rng=self._rng, **r1)
        self.mechanism2 = MLPMechanism.random(self.graph, hidden=self.hidden, rng=self._rng, **r2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, burn_in: int = 100) -> dict:
        """Generate the two-regime dataset.

        Same contract as :class:`NlinearSCMT`, plus ``"mechanism_regime2"``
        and ``"switch_t"`` (index into the *observed* ``(T,)`` window) —
        see the class docstring's "downstream single-mechanism contract"
        note for how ``"mechanism"`` (regime 1) should be interpreted.
        """
        total_T = self.T + burn_in
        switch_t_abs = burn_in + round(self.switch_frac * self.T)
        X_full = np.zeros((self.N, total_T, self.k))

        self._roll(X_full, np.arange(self.N), total_T, switch_t_abs)
        for _ in range(self.max_resample):
            bad = self._diverging(X_full)
            if not bad.any():
                break
            self._roll(X_full, np.nonzero(bad)[0], total_T, switch_t_abs)
        if self._diverging(X_full).any():
            np.clip(X_full, -self.clip, self.clip, out=X_full)

        X = X_full[:, burn_in:, :]  # shape (N, T, k)

        latent_final = X[:, -1, 0]
        threshold = float(np.median(latent_final))
        Y = (latent_final > threshold).astype(int)

        return {
            "X": X,
            "Y": Y,
            "graph": self.graph,
            "mechanism": self.mechanism1,
            "mechanism_regime2": self.mechanism2,
            "switch_t": switch_t_abs - burn_in,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _roll(self, X_full: np.ndarray, rows: np.ndarray, total_T: int, switch_t_abs: int) -> None:
        """Simulate the forward dynamics for ``rows`` in place, switching
        from ``mechanism1`` to ``mechanism2`` at absolute time ``switch_t_abs``."""
        rows = np.sort(np.asarray(rows))
        n = int(rows.size)
        if n == 0:
            return
        sub = np.zeros((n, total_T, self.k))
        for lag in range(self.L):
            sub[:, lag, :] = self._sample_noise(n, self.k) * 0.1
        noise = self._sample_noise(n * total_T * self.k).reshape(n, total_T, self.k)
        for t in range(self.L, total_T):
            window = sub[:, t - self.L : t, :]  # (n, L, k), oldest→newest
            mech = self.mechanism1 if t < switch_t_abs else self.mechanism2
            x_t = noise[:, t, :].copy()
            x_t += mech.forward_numpy(window)
            sub[:, t, :] = x_t
        X_full[rows] = sub

    def _diverging(self, X_full: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)`` mask of trajectories that blew up or went non-finite."""
        finite = np.isfinite(X_full).all(axis=(1, 2))
        bounded = np.abs(np.nan_to_num(X_full, nan=np.inf)).max(axis=(1, 2)) <= self.clip
        return ~(finite & bounded)

    def _sample_noise(self, *shape) -> np.ndarray:
        """Draw noise from the configured innovation distribution (as linear)."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        elif self.noise_type == "uniform":
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1
        else:  # gaussian (M4/H5 negative-control ablation)
            return self._rng.normal(loc=0.0, scale=_GAUSSIAN_STD, size=size)


#: Third regime's defaults for the R=3 HMM preset -- an *intermediate*
#: dynamics between regime 1 (:data:`_REGIME1_DEFAULTS`, most persistent) and
#: regime 2 (:data:`_REGIME2_DEFAULTS`, most contractive), so the three
#: regimes are ordered and deterministically distinguishable by ``decay``.
#: Its ``decay_range`` sits strictly between the other two with a safety gap.
_REGIME3_DEFAULTS: dict = {
    "decay_range": (0.15, 0.28),
    "gain": 0.55,
    "spectral_cap": 0.7,
    "init_gain": 0.5,
    "activation": "tanh",
}


class HMMRegimeSwitchNlinearSCMT:
    """Hidden-Markov regime-switching extension of :class:`NlinearSCMT` (M4/H7).

    Unlike :class:`RegimeSwitchNlinearSCMT` (two regimes, a single
    *deterministic* switch at T/2), this generator draws a genuine **hidden
    Markov regime path** per sequence: at each timestep the active regime is
    sampled from an ``R x R`` transition matrix with a strong self-transition
    probability (:attr:`p_stay`), so regimes *persist* and structural breaks
    occur at **random change-points** — a different number and placement of
    breaks per trajectory. ``R in {2, 3}`` regimes are supported (per
    ``docs/updated_general_plan.md`` §Benchmarks, NlinearSCM-T
    regime-switching ablation).

    All regimes share the *same* causal graph (:func:`_sample_graph`); only
    the per-node MLP mechanism's numeric parameters differ between regimes —
    a **parameter** regime switch, not a graph change, exactly as in
    :class:`RegimeSwitchNlinearSCMT`.

    Regime 0 (:attr:`mechanisms` ``[0]``) is drawn as the **first**
    :meth:`~causaltemp_xai.benchmarks.mechanisms.MLPMechanism.random` call
    after the graph, so — given the same seed, graph, and ``hidden`` — it is
    bit-identical to a plain :class:`NlinearSCMT`. This isolates the ablation
    to exactly the HMM structure. The regime path is drawn from an
    **independent** RNG stream (seeded ``[seed, 777]``) so the mechanism
    weights are unaffected by the path sampling.

    Downstream single-mechanism contract
    -------------------------------------
    Identical in spirit to :class:`RegimeSwitchNlinearSCMT`: CF-faith,
    oracle structural-CF, and CARLA-style recourse all assume **one**
    time-invariant mechanism per dataset. :meth:`generate` therefore returns
    regime 0 as the dataset's single ``"mechanism"`` — the *nominal* model —
    with the full regime set (``"mechanisms"``), the ``"transition_matrix"``,
    and the per-sequence ground-truth ``"regime_path"`` (shape ``(N, T)``)
    returned as diagnostic metadata, **not** wired into CF-faith/oracle-CF.
    CF-faith scores on this preset therefore measure faithfulness against the
    nominal (regime-0) mechanism, not literally-correct ground truth at
    timesteps the chain has switched away from regime 0 — this mismatch is
    the deliberate H7 stress condition, not a bug, and means CF-faith numbers
    on this preset are not directly comparable to other presets'.

    Parameters
    ----------
    k, L, sparsity, noise_type, T, N, seed:
        As in :class:`NlinearSCMT`.
    hidden:
        Hidden width of each regime's per-node MLP (shared across regimes).
    n_regimes:
        Number of regimes ``R in {2, 3}``.
    p_stay:
        Self-transition probability of the HMM (diagonal of the transition
        matrix). Must be in ``(0, 1)``; default ``0.9`` (regimes persist for
        ~10 steps on average, giving a handful of change-points over a
        ``T``-step observed window). The off-diagonal mass ``1 - p_stay`` is
        split uniformly over the other ``R - 1`` regimes.
    regimes:
        Optional explicit list of ``R`` hyperparameter dicts forwarded to
        :meth:`MLPMechanism.random`. Defaults to the first ``R`` of
        (:data:`_REGIME1_DEFAULTS`, :data:`_REGIME2_DEFAULTS`,
        :data:`_REGIME3_DEFAULTS`), which have non-overlapping ``decay_range``
        so the regimes are deterministically, not merely statistically,
        distinguishable.
    clip, max_resample:
        As in :class:`NlinearSCMT`.
    """

    def __init__(
        self,
        k: int = 5,
        L: int = 2,
        sparsity: float = 0.3,
        noise_type: Literal["laplace", "uniform", "gaussian"] = "laplace",
        T: int = 50,
        N: int = 200,
        seed: Optional[int] = 42,
        hidden: int = 16,
        n_regimes: int = 3,
        p_stay: float = 0.9,
        regimes: list[dict] | None = None,
        clip: float = 1e3,
        max_resample: int = 10,
    ) -> None:
        if noise_type not in _NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {_NOISE_TYPES}, got {noise_type!r}")
        if n_regimes not in (2, 3):
            raise ValueError(f"n_regimes must be 2 or 3, got {n_regimes!r}")
        if not (0.0 < p_stay < 1.0):
            raise ValueError(f"p_stay must be in (0, 1), got {p_stay!r}")
        self.k = k
        self.L = L
        self.sparsity = sparsity
        self.noise_type = noise_type
        self.T = T
        self.N = N
        self.seed = seed
        self.hidden = hidden
        self.n_regimes = n_regimes
        self.p_stay = p_stay
        self.clip = clip
        self.max_resample = max_resample
        if regimes is None:
            defaults = (_REGIME1_DEFAULTS, _REGIME2_DEFAULTS, _REGIME3_DEFAULTS)
            hparams = [dict(defaults[r]) for r in range(n_regimes)]
        else:
            if len(regimes) != n_regimes:
                raise ValueError(
                    f"regimes must have length n_regimes={n_regimes}, got {len(regimes)}"
                )
            hparams = [dict(r) for r in regimes]
        self.regime_hparams = hparams
        self._rng = np.random.default_rng(seed)
        # Independent stream for the regime path so mechanism weights (drawn
        # from self._rng) stay bit-identical to a plain NlinearSCMT for
        # regime 0. See class docstring.
        self._path_rng = np.random.default_rng([seed if seed is not None else 0, 777])
        # One graph, shared by all regimes; sampled before any mechanism
        # weights -- mirrors NlinearSCMT / RegimeSwitchNlinearSCMT build order.
        self.graph = _sample_graph(self.k, self.L, self.sparsity, self._rng)
        self.mechanisms = [
            MLPMechanism.random(self.graph, hidden=self.hidden, rng=self._rng, **hp)
            for hp in hparams
        ]
        self.transition_matrix = self._build_transition_matrix()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, burn_in: int = 100) -> dict:
        """Generate the HMM regime-switching dataset.

        Same core contract as :class:`NlinearSCMT` (``X``, ``Y``, ``graph``,
        ``mechanism``), plus regime-switching diagnostics: ``"mechanisms"``
        (the full ``R``-element list), ``"transition_matrix"`` (``(R, R)``),
        and ``"regime_path"`` (``(N, T)`` int array of the active regime at
        each *observed* timestep). See the class docstring's downstream
        single-mechanism contract for how ``"mechanism"`` (regime 0) should
        be interpreted.
        """
        total_T = self.T + burn_in
        # Full regime path over burn-in + observed window, deterministic
        # given the seed and independent of the noise/weight draws.
        self._regime_path_full = self._sample_regime_path(self.N, total_T)
        X_full = np.zeros((self.N, total_T, self.k))

        self._roll(X_full, np.arange(self.N), total_T)
        for _ in range(self.max_resample):
            bad = self._diverging(X_full)
            if not bad.any():
                break
            self._roll(X_full, np.nonzero(bad)[0], total_T)
        if self._diverging(X_full).any():
            np.clip(X_full, -self.clip, self.clip, out=X_full)

        X = X_full[:, burn_in:, :]  # shape (N, T, k)
        regime_path = self._regime_path_full[:, burn_in:]  # (N, T)

        latent_final = X[:, -1, 0]
        threshold = float(np.median(latent_final))
        Y = (latent_final > threshold).astype(int)

        return {
            "X": X,
            "Y": Y,
            "graph": self.graph,
            "mechanism": self.mechanisms[0],
            "mechanisms": self.mechanisms,
            "transition_matrix": self.transition_matrix,
            "regime_path": regime_path,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> np.ndarray:
        """``(R, R)`` row-stochastic matrix: ``p_stay`` on the diagonal, the
        remaining ``1 - p_stay`` mass split uniformly over other regimes."""
        R = self.n_regimes
        off = (1.0 - self.p_stay) / (R - 1)
        P = np.full((R, R), off)
        np.fill_diagonal(P, self.p_stay)
        return P

    def _sample_regime_path(self, n: int, total_T: int) -> np.ndarray:
        """Sample an ``(n, total_T)`` integer regime path from the HMM.

        Every sequence starts in regime 0 (so the burn-in settles from the
        nominal dynamics), then transitions each step per
        :attr:`transition_matrix`. Vectorised over sequences.
        """
        path = np.zeros((n, total_T), dtype=int)
        cumP = np.cumsum(self.transition_matrix, axis=1)  # (R, R)
        for t in range(1, total_T):
            u = self._path_rng.random(n)
            cur = path[:, t - 1]
            # Next regime = first column whose cumulative prob exceeds u,
            # per the current regime's transition row.
            path[:, t] = (u[:, None] >= cumP[cur]).sum(axis=1)
        np.clip(path, 0, self.n_regimes - 1, out=path)
        return path

    def _roll(self, X_full: np.ndarray, rows: np.ndarray, total_T: int) -> None:
        """Simulate the forward dynamics for ``rows`` in place, applying each
        timestep's per-sequence active regime mechanism (grouped by regime)."""
        rows = np.sort(np.asarray(rows))
        n = int(rows.size)
        if n == 0:
            return
        path = self._regime_path_full[rows]  # (n, total_T)
        sub = np.zeros((n, total_T, self.k))
        for lag in range(self.L):
            sub[:, lag, :] = self._sample_noise(n, self.k) * 0.1
        noise = self._sample_noise(n * total_T * self.k).reshape(n, total_T, self.k)
        for t in range(self.L, total_T):
            window = sub[:, t - self.L : t, :]  # (n, L, k), oldest→newest
            x_t = noise[:, t, :].copy()
            reg_t = path[:, t]  # (n,) active regime per sequence at time t
            out = np.zeros((n, self.k))
            for r in range(self.n_regimes):
                mask = reg_t == r
                if mask.any():
                    out[mask] = self.mechanisms[r].forward_numpy(window[mask])
            sub[:, t, :] = x_t + out
        X_full[rows] = sub

    def _diverging(self, X_full: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)`` mask of trajectories that blew up or went non-finite."""
        finite = np.isfinite(X_full).all(axis=(1, 2))
        bounded = np.abs(np.nan_to_num(X_full, nan=np.inf)).max(axis=(1, 2)) <= self.clip
        return ~(finite & bounded)

    def _sample_noise(self, *shape) -> np.ndarray:
        """Draw noise from the configured innovation distribution (as linear)."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        elif self.noise_type == "uniform":
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1
        else:  # gaussian (M4/H5 negative-control ablation)
            return self._rng.normal(loc=0.0, scale=_GAUSSIAN_STD, size=size)


def _stabilise(A: np.ndarray, target_radius: float = 0.9) -> np.ndarray:
    """Rescale *A* so its spectral radius is at most *target_radius*."""
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius > 1e-8:
        A = A * (target_radius / radius)
    return A
