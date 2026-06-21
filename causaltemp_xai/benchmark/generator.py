"""Structural-causal-model generators for temporal data.

* :class:`LinearSCMT` — VAR(L) process; each variable ``x_i`` at time ``t`` is a
  linear function of its causal parents at lags 1…L plus non-Gaussian noise::

      x_t  =  sum_{l=1}^{L}  A_l @ x_{t-l}  +  eps_t

* :class:`NlinearSCMT` — same graph machinery and noise, but the linear
  mechanism is swapped for an additive-noise per-node MLP
  (:class:`~causaltemp_xai.benchmark.mechanisms.MLPMechanism`)::

      x_t  =  mechanism.forward_numpy(window_{t-L..t-1})  +  eps_t

Both share the graph sampler (:func:`_sample_graph`) so the no-self-loop-at-lag-1
rule and the per-lag Bernoulli draw are identical, and ``eps_t`` is drawn from
either a Laplace or Uniform distribution.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from causaltemp_xai.benchmark.mechanisms import LinearMechanism, MLPMechanism


_NOISE_TYPES = ("laplace", "uniform")


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


def _sample_graph(
    k: int, L: int, sparsity: float, rng: np.random.Generator
) -> np.ndarray:
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
        Noise distribution for the innovation terms.  Either ``"laplace"`` or
        ``"uniform"``.
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
        noise_type: Literal["laplace", "uniform"] = "laplace",
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
        noise = self._sample_noise(self.N * total_T * self.k).reshape(
            self.N, total_T, self.k
        )
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
        """Draw noise samples from the configured non-Gaussian distribution."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        else:  # uniform
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1


class NlinearSCMT:
    """Nonlinear SCM-T generator: additive-noise per-node MLP transitions.

    Shares :class:`LinearSCMT`'s graph machinery, noise distributions, burn-in and
    median-threshold label rule, but the linear ``A_l @ x`` mechanism is replaced
    by an :class:`~causaltemp_xai.benchmark.mechanisms.MLPMechanism`. The forward
    step is::

        x_t = mechanism.forward_numpy(window_{t-L..t-1}) + eps_t

    The mechanism is contractive (leaky ``decay`` + bounded ``gain·tanh`` branch
    with spectral-norm-capped weights), so trajectories stay bounded over long
    horizons; a deterministic divergence-resample guard (then a clip fallback)
    catches the rare blow-up without breaking ``same seed ⇒ identical X``.

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
        Hidden-layer activation (only ``"tanh"`` supported).
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
        noise_type: Literal["laplace", "uniform"] = "laplace",
        T: int = 50,
        N: int = 200,
        seed: Optional[int] = 42,
        hidden: int = 16,
        gain: float = 0.8,
        decay_range: tuple[float, float] = (0.3, 0.8),
        spectral_cap: float = 0.9,
        init_gain: float = 0.7,
        activation: str = "tanh",
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
        """Draw noise from the configured non-Gaussian distribution (as linear)."""
        size = shape if len(shape) > 1 else shape[0]
        if self.noise_type == "laplace":
            return self._rng.laplace(loc=0.0, scale=0.1, size=size)
        else:  # uniform
            return self._rng.uniform(low=-0.17, high=0.17, size=size)  # std ≈ 0.1


def _stabilise(A: np.ndarray, target_radius: float = 0.9) -> np.ndarray:
    """Rescale *A* so its spectral radius is at most *target_radius*."""
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius > 1e-8:
        A = A * (target_radius / radius)
    return A
