"""Polymorphic transition mechanisms for temporal SCMs.

A :class:`Mechanism` maps a lag *window* of recent states to the
**deterministic next-step mean** (pre-noise). It is the single abstraction every
``A @ x`` call site routes through, so the same SCM can be evaluated in numpy
(generation, CF-faith) and in differentiable torch (CARLA), and serialized to
disk — without any site hard-coding linearity.

Window contract
---------------
Callers always pass exactly ``L`` rows, ordered oldest→newest, zero-padding the
oldest rows when fewer than ``L`` are available. Concretely ``history[..., -1, :]``
is lag 1 (``x_{t-1}``) and ``history[..., -L, :]`` is lag ``L`` (``x_{t-L}``). A
lag that would reach before ``t=0`` is therefore a zero row and contributes
nothing — preserving the ``if lag_t >= 0`` guard the original ``cf_faith`` /
``carla`` loops used. This is a no-op for ``L=1`` but is the correctness landmine
for ``L>1`` with an early ``intervention_t``.

Shapes: ``forward_*`` accepts a single window ``(L, k)`` → returns ``(k,)``, or a
batched window ``(N, L, k)`` → returns ``(N, k)``.

Serialization contract (npz-coercion landmine)
----------------------------------------------
``state_dict`` is persisted via ``np.savez(**d)`` and reloaded via
``dict(np.load(...))``. ``np.load`` returns **every** value as an ``np.ndarray``,
so scalar/string fields come back as 0-d arrays (``d["__type__"]`` becomes
``array("linear")``, ``d["k"]`` becomes ``array(3)``). ``from_state_dict`` /
:func:`mechanism_from_state_dict` therefore **coerce** non-array fields
(``str(...)`` / ``int(...)`` / ``float(...)``) before use.

Scope
-----
:class:`MLPMechanism` is an **additive-noise** transition (the noise is added
*after* the deterministic mean, never inside ``forward_*``). That additivity is
what keeps Pearl abduction an exact subtraction downstream. Nonlinear *mixing*
``x = g(z)`` and *non-additive* noise are out of scope this iteration — see the
:mod:`causaltemp_xai.benchmarks.generator` module docstring and the plan Backlog.
"""

from __future__ import annotations

import numpy as np

try:  # torch is a hard dep elsewhere; import lazily-friendly for clarity
    import torch
except Exception:  # pragma: no cover - torch always present in this project
    torch = None  # type: ignore


def lag_window(arr: np.ndarray, t: int, L: int, k: int) -> np.ndarray:
    """Build the ``(L, k)`` lag window feeding a mechanism at time ``t``.

    Rows are ordered oldest→newest (``window[-1]`` is lag 1, ``x_{t-1}``); any
    row that would reach before ``t=0`` is left zero, replicating the original
    ``if lag_t >= 0`` guard (a zero row contributes nothing). This is the single
    source of truth for the window contract documented in the module docstring;
    ``cf_faith`` and ``structural_cf`` both import it so they cannot drift.
    """
    window = np.zeros((L, k))
    for j in range(L):
        src = t - L + j  # row position j maps to absolute time `src`
        if src >= 0:
            window[j] = arr[src]
    return window


class Mechanism:
    """Abstract transition mechanism.

    Subclasses implement :meth:`forward_numpy` / :meth:`forward_torch` (the
    deterministic next-step mean over a lag window) and round-trippable
    :meth:`state_dict` / :meth:`from_state_dict` with a ``"__type__"``
    discriminator.

    Attributes
    ----------
    k : int
        Number of variables.
    L : int
        Maximum lag (window length the mechanism consumes).
    """

    k: int
    L: int

    def forward_numpy(self, history: np.ndarray) -> np.ndarray:
        """Deterministic next-step mean from a ``(L, k)`` or ``(N, L, k)`` window."""
        raise NotImplementedError

    def forward_torch(self, history):
        """Differentiable next-step mean from a ``(L, k)`` or ``(N, L, k)`` window."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        """Round-trippable serialization including a ``"__type__"`` discriminator."""
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, d: dict) -> Mechanism:
        """Reconstruct a mechanism from :meth:`state_dict` output.

        Implementations must coerce non-array scalar/string fields (which
        ``np.load`` returns as 0-d arrays) — see the module docstring.
        """
        raise NotImplementedError


class LinearMechanism(Mechanism):
    """VAR(L) mechanism: ``x_t = sum_l A_l @ x_{t-l-1}``.

    Parameters
    ----------
    A_list:
        List of ``L`` coefficient matrices, each ``(k, k)``; ``A_list[l]`` acts on
        lag ``l+1``.
    """

    def __init__(self, A_list: list[np.ndarray]) -> None:
        if len(A_list) == 0:
            raise ValueError("A_list must contain at least one matrix")
        self.A_list = [np.asarray(A, dtype=float) for A in A_list]
        self.L = len(self.A_list)
        self.k = self.A_list[0].shape[0]

    # ------------------------------------------------------------------
    # Forward evaluation
    # ------------------------------------------------------------------

    def forward_numpy(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if history.ndim == 2:  # single window (L, k) -> (k,)
            # Accumulate `A_l @ x_{t-l-1}` in lag order — bit-identical to the
            # original cf_faith loop (zero-padded rows contribute 0).
            result = np.zeros(self.k)
            for l, A in enumerate(self.A_list):
                result += A @ history[-(l + 1), :]
            return result
        # batched window (N, L, k) -> (N, k); `x @ A.T` matches the generator.
        result = np.zeros((history.shape[0], self.k))
        for l, A in enumerate(self.A_list):
            result += history[:, -(l + 1), :] @ A.T
        return result

    def forward_torch(self, history):
        if history.ndim == 2:  # (L, k) -> (k,)
            acc = torch.zeros(self.k, dtype=history.dtype)
            for l in range(self.L):
                A = torch.as_tensor(self.A_list[l], dtype=history.dtype)
                acc = acc + A @ history[-(l + 1), :]
            return acc
        # (N, L, k) -> (N, k)
        acc = torch.zeros(history.shape[0], self.k, dtype=history.dtype)
        for l in range(self.L):
            A = torch.as_tensor(self.A_list[l], dtype=history.dtype)
            acc = acc + history[:, -(l + 1), :] @ A.T
        return acc

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d: dict = {"__type__": "linear", "k": int(self.k), "L": int(self.L)}
        for l, A in enumerate(self.A_list):
            d[f"A_{l}"] = np.asarray(A, dtype=float)
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> LinearMechanism:
        # Restore lag order by sorting A_<l> keys (mirror data_io.load_dataset).
        a_keys = sorted(
            (key for key in d if str(key).startswith("A_")),
            key=lambda s: int(str(s).split("_")[1]),
        )
        A_list = [np.asarray(d[key], dtype=float) for key in a_keys]
        return cls(A_list)


#: Hidden-layer activation registry. ``"tanh"`` is the original monotonic
#: choice; ``"nonmonotonic"`` (``np.sin``) is the M4/H6 ablation -- bounded in
#: ``[-1, 1]`` and 1-Lipschitz (``|cos| <= 1``) just like ``tanh``, so it
#: shares tanh's boundedness/Lipschitz properties exactly, but is genuinely
#: non-monotonic (a local max at ``pi/2``, unlike strictly-increasing
#: ``tanh``) -- see ``docs/archive/m4_ablation_presets_smoke.md``.
_ACTIVATIONS_NP = {"tanh": np.tanh, "nonmonotonic": np.sin}


def _spectral_cap(W: np.ndarray, s: float) -> np.ndarray:
    """Rescale a 2-D matrix so its spectral norm (largest singular value) ≤ ``s``.

    One-shot rescale by ``min(1, s / sigma_max)`` — analogous to
    :func:`generator._stabilise` but using the operator 2-norm, which bounds the
    layer's Lipschitz constant (Rhino / CausalDynamics stability recipe).
    """
    sigma = float(np.linalg.norm(W, ord=2)) if W.size else 0.0
    if sigma > s and sigma > 1e-12:
        W = W * (s / sigma)
    return W


class MLPMechanism(Mechanism):
    """Per-node additive-noise MLP transition with contractive dynamics.

    The deterministic next-step mean for node ``i`` is::

        mean_i(history) = decay_i · x_{t-1}^i + gain · tanh( MLP_i( masked lags ) )

    where ``MLP_i`` is a 2-layer MLP over node ``i``'s masked lagged parents.
    The ``k`` per-node MLPs are stacked into batched weight tensors so the
    forward pass is a single batched op (no Python loop over nodes).

    Parameters
    ----------
    graph:
        Binary adjacency tensor ``(k, k, L)``; ``graph[i, j, l]`` is 1 if
        variable *j* causes variable *i* at lag ``l+1``. Defines the per-node
        parent mask.
    hidden:
        Hidden width ``H`` of each per-node MLP.
    decay:
        Leaky decay coefficients, shape ``(k,)``; ``decay_i`` scales lag-1
        ``x_{t-1}^i``.
    gain:
        Scalar multiplier on the bounded ``tanh`` branch.
    W1, b1, W2, b2:
        Stacked weights ``W1 (k, H, k*L)``, ``b1 (k, H)``, ``W2 (k, 1, H)``,
        ``b2 (k, 1)``.
    activation:
        Hidden-layer activation: ``"tanh"`` (default, monotonic) or
        ``"nonmonotonic"`` (M4/H6 ablation -- ``sin``, bounded in ``[-1, 1]``
        and 1-Lipschitz like ``tanh``, but not monotonic; see
        ``docs/archive/m4_ablation_presets_smoke.md``). The **output** branch is
        always ``tanh`` regardless of this choice, so the mechanism's global
        boundedness guarantee is unaffected by which hidden activation is
        selected -- only the hidden representation's monotonicity varies.

    Notes
    -----
    The masked input of each node is scaled by ``1/√(max(n_active_parents, 1))``
    so pre-activations land in ``tanh``'s curved region (genuine nonlinearity).
    The ``max(·, 1)`` guard makes a 0-parent node reduce to pure decay
    (``mean_i = decay_i · x_{t-1}^i``) instead of dividing by zero.
    """

    def __init__(
        self,
        graph: np.ndarray,
        hidden: int,
        decay: np.ndarray,
        gain: float,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        activation: str = "tanh",
    ) -> None:
        if activation not in _ACTIVATIONS_NP:
            raise ValueError(
                f"activation must be one of {tuple(_ACTIVATIONS_NP)}, got {activation!r}"
            )
        self.graph = np.asarray(graph, dtype=float)
        self.k, _, self.L = self.graph.shape
        self.hidden = int(hidden)
        self.gain = float(gain)
        self.activation = activation
        self.decay = np.asarray(decay, dtype=float).reshape(self.k)
        self.W1 = np.asarray(W1, dtype=float)
        self.b1 = np.asarray(b1, dtype=float)
        self.W2 = np.asarray(W2, dtype=float)
        self.b2 = np.asarray(b2, dtype=float)

        # Parent mask flattened over (parent j, lag l) with C-order index j*L + l,
        # matching the flattened history vector built in `_flatten`.
        self.parent_mask = self.graph.reshape(self.k, self.k * self.L)
        n_active = self.parent_mask.sum(axis=1)
        input_scale = 1.0 / np.sqrt(np.maximum(n_active, 1.0))
        # Precompute mask × per-node input scale once → applied to flat history.
        self._masked_weight = self.parent_mask * input_scale[:, None]
        self._torch_cache: dict = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        graph: np.ndarray,
        hidden: int,
        rng: np.random.Generator,
        decay_range: tuple[float, float] = (0.3, 0.8),
        gain: float = 0.8,
        spectral_cap: float = 0.9,
        init_gain: float = 0.7,
        activation: str = "tanh",
    ) -> MLPMechanism:
        """Sample + stabilize a random per-node MLP mechanism.

        Weights use ``std = init_gain · √(1/fan_in)`` (bias 0) and each per-node
        weight matrix is spectral-norm-capped at ``spectral_cap`` (Lipschitz < 1
        for the nonlinear branch).
        """
        graph = np.asarray(graph, dtype=float)
        k, _, L = graph.shape
        d_in = k * L

        std1 = init_gain * np.sqrt(1.0 / d_in)
        std2 = init_gain * np.sqrt(1.0 / hidden)
        W1 = rng.normal(0.0, std1, size=(k, hidden, d_in))
        W2 = rng.normal(0.0, std2, size=(k, 1, hidden))
        # Spectral-norm cap each per-node weight matrix (one-shot rescale).
        for i in range(k):
            W1[i] = _spectral_cap(W1[i], spectral_cap)
            W2[i] = _spectral_cap(W2[i], spectral_cap)
        b1 = np.zeros((k, hidden))
        b2 = np.zeros((k, 1))
        decay = rng.uniform(decay_range[0], decay_range[1], size=k)
        return cls(
            graph=graph,
            hidden=hidden,
            decay=decay,
            gain=gain,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            activation=activation,
        )

    # ------------------------------------------------------------------
    # Forward evaluation
    # ------------------------------------------------------------------

    def _flatten_numpy(self, history: np.ndarray) -> np.ndarray:
        """Flatten an ``(N, L, k)`` window to masked per-node inputs ``(N, k, k*L)``.

        ``flat[n, j*L + l]`` is variable ``j`` at lag ``l+1`` (``history[n, -(l+1), j]``).
        """
        # Reverse lag axis (oldest→newest ⇒ lag L…1), then (l, j)→(j, l) and flatten.
        rev = history[:, ::-1, :]  # (N, L, k): rev[n, l, j] = x at lag l+1
        flat = np.ascontiguousarray(rev.transpose(0, 2, 1)).reshape(
            history.shape[0], self.k * self.L
        )  # (N, k*L), index j*L + l
        # Broadcast the shared flat history against each node's mask × input scale.
        return flat[:, None, :] * self._masked_weight[None, :, :]  # (N, k, k*L)

    def forward_numpy(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        single = history.ndim == 2
        if single:
            history = history[None]  # (1, L, k)
        masked = self._flatten_numpy(history)  # (N, k, k*L)
        act = _ACTIVATIONS_NP[self.activation]
        h = act(np.einsum("khd,nkd->nkh", self.W1, masked) + self.b1)  # (N, k, H)
        # Add output bias BEFORE squeezing the singleton output dim.
        out = (np.einsum("koh,nkh->nko", self.W2, h) + self.b2)[..., 0]  # (N, k)
        result = self.decay * history[:, -1, :] + self.gain * np.tanh(out)  # (N, k)
        return result[0] if single else result

    def _torch_weights(self, like):
        key = (like.dtype, like.device)
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = {
                name: torch.as_tensor(arr, dtype=like.dtype, device=like.device)
                for name, arr in (
                    ("W1", self.W1),
                    ("b1", self.b1),
                    ("W2", self.W2),
                    ("b2", self.b2),
                    ("decay", self.decay),
                    ("masked_weight", self._masked_weight),
                )
            }
            self._torch_cache[key] = cached
        return cached

    def forward_torch(self, history):
        single = history.ndim == 2
        if single:
            history = history.unsqueeze(0)  # (1, L, k)
        w = self._torch_weights(history)
        rev = torch.flip(history, dims=[1])  # (N, L, k): lag L…1
        flat = rev.transpose(1, 2).reshape(history.shape[0], self.k * self.L)
        masked = flat.unsqueeze(1) * w["masked_weight"].unsqueeze(0)  # (N, k, k*L)
        if self.activation == "tanh":
            h = torch.tanh(torch.einsum("khd,nkd->nkh", w["W1"], masked) + w["b1"])
        elif self.activation == "nonmonotonic":
            h = torch.sin(torch.einsum("khd,nkd->nkh", w["W1"], masked) + w["b1"])
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unsupported activation {self.activation!r}")
        out = (torch.einsum("koh,nkh->nko", w["W2"], h) + w["b2"])[..., 0]  # (N, k)
        result = w["decay"] * history[:, -1, :] + self.gain * torch.tanh(out)
        return result.squeeze(0) if single else result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "__type__": "mlp",
            "k": int(self.k),
            "L": int(self.L),
            "hidden": int(self.hidden),
            "gain": float(self.gain),
            "activation": self.activation,
            "graph": np.asarray(self.graph, dtype=float),
            "decay": np.asarray(self.decay, dtype=float),
            "W1": np.asarray(self.W1, dtype=float),
            "b1": np.asarray(self.b1, dtype=float),
            "W2": np.asarray(self.W2, dtype=float),
            "b2": np.asarray(self.b2, dtype=float),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> MLPMechanism:
        # np.load returns scalars/strings as 0-d arrays — coerce them.
        activation = d["activation"]
        if not isinstance(activation, str):
            activation = str(np.asarray(activation).item())
        return cls(
            graph=np.asarray(d["graph"], dtype=float),
            hidden=int(np.asarray(d["hidden"]).item()),
            decay=np.asarray(d["decay"], dtype=float),
            gain=float(np.asarray(d["gain"]).item()),
            W1=np.asarray(d["W1"], dtype=float),
            b1=np.asarray(d["b1"], dtype=float),
            W2=np.asarray(d["W2"], dtype=float),
            b2=np.asarray(d["b2"], dtype=float),
            activation=activation,
        )


class SpringMechanism(Mechanism):
    """Additive-noise spring-coupled particle system (M4c, non-dissipative).

    ``n_particles`` particles, each exposed as **two** channels (position,
    velocity), so ``k = 2 * n_particles``, ``L=1``. Channel layout:
    ``history[..., 0:n_particles]`` = positions, ``history[..., n_particles:]``
    = velocities. One symplectic-Euler step per timestep::

        a_i(x_{t-1})   =  -sum_j K[i, j] * (x_{t-1}^i - x_{t-1}^j)   (graph neighbors j)
        mean(v_t^i)    =  v_{t-1}^i + dt * a_i(x_{t-1})
        mean(x_t^i)    =  x_{t-1}^i + dt * v_{t-1}^i               (position uses the
                                                                      *previous* velocity,
                                                                      per symplectic Euler)

    **Velocity is a genuine exposed state channel, not reconstructed by
    finite-differencing noisy position history.** An earlier design (``L=2``,
    ``v ~= (x_{t-1}-x_{t-2})/dt``) was rejected after a smoke-scale test
    showed it amplifies each step's additive noise by ``1/dt`` and compounds
    over the trajectory -- a controlled test isolated this precisely: the
    deterministic integrator alone stayed bounded (``max|x| = 0.01`` over 130
    steps from a ``0.01`` initial kick), but adding per-step ``Laplace(0,
    0.1)`` noise blew the same run up to ``max|x| > 160`` by step 129. This
    design avoids that failure mode entirely: each channel's own noise is
    added once per step, never re-differentiated.

    Deliberately **no damping term** -- the coupling is a directed SCM
    influence (``graph[vel_i, pos_j, 0]`` = "particle j's position causes
    particle i's velocity"), not a literal symmetric physical spring network,
    and the only source of energy drift is symplectic Euler's own
    finite-``dt`` error. That drift is exactly what
    ``causaltemp_xai.benchmarks.diagnostics.empirical_contraction_rate`` is
    for measuring empirically (M4c DoD: "do not assume rho ~= 1").

    ``graph`` encodes only **cross-particle** coupling: ``graph[vel_i, pos_j,
    0] = 1`` for spring-neighbors ``j != i``. The internal position<-velocity
    relationship (every particle's position is driven by its own velocity)
    and each particle's own position term in its own acceleration are both
    treated as baseline/structural, not graph-worthy -- the same convention
    :class:`MLPMechanism` uses for its ``decay_i * x_{t-1}^i`` term. A
    consequence: **every** position channel has zero in-degree by
    construction (its only true dependency is its own velocity, deliberately
    excluded from `graph`), so :func:`exogenous_channels` on the full
    ``2 * n_particles``-channel graph is not directly "uninfluenced
    particle" -- only the **velocity** half of its output identifies M4c's
    "particles p4/p5" (an isolated particle's velocity has no incoming
    graph edge; every particle's position never does, non-informatively).
    See ``SpringSCMT`` for the particle-level wrapper.

    Mass is fixed at 1 for every particle (minimal viable design, matching
    this codebase's convention elsewhere of keeping ablation families to one
    or two free scalars).
    """

    def __init__(
        self,
        graph: np.ndarray,
        k_spring: float,
        dt: float,
    ) -> None:
        self.graph = np.asarray(graph, dtype=float)
        self.k, _, self.L = self.graph.shape
        if self.L != 1:
            raise ValueError(f"SpringMechanism requires L=1, got L={self.L}")
        if self.k % 2 != 0:
            raise ValueError(f"SpringMechanism requires an even k (2*n_particles), got k={self.k}")
        self.n_particles = self.k // 2
        self.k_spring = float(k_spring)
        self.dt = float(dt)
        p = self.n_particles
        # Directed coupling K[i, j]: influence of particle j's position on
        # particle i's acceleration, read from the velocity-row / position-col
        # block of `graph` (graph[p+i, j, 0] = 1 iff j couples into i).
        self.K = self.k_spring * self.graph[p : 2 * p, 0:p, 0]

    # ------------------------------------------------------------------
    # Forward evaluation
    # ------------------------------------------------------------------

    def forward_numpy(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        single = history.ndim == 2
        if single:
            history = history[None]  # (1, L, k)
        p = self.n_particles
        state = history[:, -1, :]  # (N, 2p)
        pos, vel = state[:, :p], state[:, p:]  # (N, p) each
        diff = pos[:, :, None] - pos[:, None, :]  # (N, p, p): [n,i,j] = x_i - x_j
        accel = -np.einsum("ij,nij->ni", self.K, diff)  # (N, p)
        new_pos = pos + self.dt * vel  # symplectic Euler: position uses OLD velocity
        new_vel = vel + self.dt * accel
        result = np.concatenate([new_pos, new_vel], axis=1)  # (N, 2p)
        return result[0] if single else result

    def forward_torch(self, history):
        single = history.ndim == 2
        if single:
            history = history.unsqueeze(0)
        p = self.n_particles
        K = torch.as_tensor(self.K, dtype=history.dtype, device=history.device)
        state = history[:, -1, :]
        pos, vel = state[:, :p], state[:, p:]
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (N, p, p)
        accel = -torch.einsum("ij,nij->ni", K, diff)
        new_pos = pos + self.dt * vel
        new_vel = vel + self.dt * accel
        result = torch.cat([new_pos, new_vel], dim=1)
        return result.squeeze(0) if single else result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "__type__": "spring",
            "graph": np.asarray(self.graph, dtype=float),
            "k_spring": float(self.k_spring),
            "dt": float(self.dt),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> SpringMechanism:
        return cls(
            graph=np.asarray(d["graph"], dtype=float),
            k_spring=float(np.asarray(d["k_spring"]).item()),
            dt=float(np.asarray(d["dt"]).item()),
        )


class KuramotoMechanism(Mechanism):
    """Additive-noise coupled-phase-oscillator system (M4c, non-dissipative).

    ``k`` oscillators, ``L=1`` (phase coupling needs only the current phase,
    no finite-difference issue). Standard sparse-graph Kuramoto update,
    discretized with step ``dt``::

        mean_i(theta_t) = theta_{t-1}^i + dt * (omega_i + (K/deg_i) * sum_j A[i,j] sin(theta_{t-1}^j - theta_{t-1}^i))

    summed over ``graph`` neighbors ``j`` only (not fully connected -- the
    sparse graph is what makes this an SCM rather than a mean-field model).
    ``deg_i = max(sum_j A[i,j], 1)`` avoids dividing by zero for a
    zero-in-degree oscillator, and is a no-op there anyway since the sum
    itself is zero (a "pacemaker" -- M4c's o4/o5 -- evolves purely at its own
    ``omega_i`` plus noise).

    Phase is tracked **unwrapped** (a plain, unbounded real number, never
    reduced ``mod 2*pi``) as the SCM state. This is the design choice that
    keeps Pearl abduction exact regardless of periodicity: ``eps = x_t -
    f(parents)`` is a plain subtraction here exactly as it is for every other
    mechanism in this benchmark, so wrapping (if ever wanted) is purely a
    downstream reporting/plotting concern and must never be baked into the
    generative model itself -- doing so would make ``eps`` ambiguous modulo
    ``2*pi`` and break exact abduction.
    """

    def __init__(
        self,
        graph: np.ndarray,
        omega: np.ndarray,
        k_coupling: float,
        dt: float,
    ) -> None:
        self.graph = np.asarray(graph, dtype=float)
        self.k, _, self.L = self.graph.shape
        if self.L != 1:
            raise ValueError(f"KuramotoMechanism requires L=1, got L={self.L}")
        self.omega = np.asarray(omega, dtype=float).reshape(self.k)
        self.k_coupling = float(k_coupling)
        self.dt = float(dt)
        A = self.graph[:, :, 0]  # (k, k): A[i, j] = 1 if j couples into i
        deg = np.maximum(A.sum(axis=1), 1.0)  # (k,)
        self.A = A
        self.inv_deg = 1.0 / deg  # (k,)

    # ------------------------------------------------------------------
    # Forward evaluation
    # ------------------------------------------------------------------

    def forward_numpy(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        single = history.ndim == 2
        if single:
            history = history[None]  # (1, L, k)
        theta = history[:, -1, :]  # (N, k)
        # sin(theta_j - theta_i) for every (i, j) pair, masked by A.
        diff = theta[:, None, :] - theta[:, :, None]  # (N, k, k): [n,i,j] = theta_j - theta_i
        coupling = np.einsum("ij,nij->ni", self.A, np.sin(diff)) * self.inv_deg  # (N, k)
        result = theta + self.dt * (self.omega + self.k_coupling * coupling)
        return result[0] if single else result

    def forward_torch(self, history):
        single = history.ndim == 2
        if single:
            history = history.unsqueeze(0)
        A = torch.as_tensor(self.A, dtype=history.dtype, device=history.device)
        inv_deg = torch.as_tensor(self.inv_deg, dtype=history.dtype, device=history.device)
        omega = torch.as_tensor(self.omega, dtype=history.dtype, device=history.device)
        theta = history[:, -1, :]
        diff = theta.unsqueeze(1) - theta.unsqueeze(2)  # (N, k, k): [n,i,j] = theta_j - theta_i
        coupling = torch.einsum("ij,nij->ni", A, torch.sin(diff)) * inv_deg
        result = theta + self.dt * (omega + self.k_coupling * coupling)
        return result.squeeze(0) if single else result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "__type__": "kuramoto",
            "graph": np.asarray(self.graph, dtype=float),
            "omega": np.asarray(self.omega, dtype=float),
            "k_coupling": float(self.k_coupling),
            "dt": float(self.dt),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> KuramotoMechanism:
        return cls(
            graph=np.asarray(d["graph"], dtype=float),
            omega=np.asarray(d["omega"], dtype=float),
            k_coupling=float(np.asarray(d["k_coupling"]).item()),
            dt=float(np.asarray(d["dt"]).item()),
        )


def mechanism_from_state_dict(d: dict) -> Mechanism:
    """Dispatch on the ``"__type__"`` discriminator to the right subclass.

    Coerces the discriminator with ``str(...)`` because ``np.load`` returns it as
    a 0-d array.
    """
    mech_type = (
        str(np.asarray(d["__type__"]).item())
        if not isinstance(d["__type__"], str)
        else d["__type__"]
    )
    if mech_type == "linear":
        return LinearMechanism.from_state_dict(d)
    if mech_type == "mlp":
        return MLPMechanism.from_state_dict(d)
    if mech_type == "spring":
        return SpringMechanism.from_state_dict(d)
    if mech_type == "kuramoto":
        return KuramotoMechanism.from_state_dict(d)
    raise ValueError(f"unknown mechanism __type__: {mech_type!r}")
