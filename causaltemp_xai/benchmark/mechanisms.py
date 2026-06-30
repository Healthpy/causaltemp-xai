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
:mod:`causaltemp_xai.benchmark.generator` module docstring and the plan Backlog.
"""

from __future__ import annotations

from typing import Union

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

    def forward_torch(self, history):  # noqa: ANN001 - torch.Tensor
        """Differentiable next-step mean from a ``(L, k)`` or ``(N, L, k)`` window."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        """Round-trippable serialization including a ``"__type__"`` discriminator."""
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, d: dict) -> "Mechanism":
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

    def forward_torch(self, history):  # noqa: ANN001
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
    def from_state_dict(cls, d: dict) -> "LinearMechanism":
        # Restore lag order by sorting A_<l> keys (mirror data_io.load_dataset).
        a_keys = sorted(
            (key for key in d if str(key).startswith("A_")),
            key=lambda s: int(str(s).split("_")[1]),
        )
        A_list = [np.asarray(d[key], dtype=float) for key in a_keys]
        return cls(A_list)


_ACTIVATIONS_NP = {"tanh": np.tanh}


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
        Hidden-layer activation (only ``"tanh"`` supported). The output branch is
        always ``tanh`` for boundedness.

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
    ) -> "MLPMechanism":
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

    def _torch_weights(self, like):  # noqa: ANN001 - torch.Tensor
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

    def forward_torch(self, history):  # noqa: ANN001 - torch.Tensor
        single = history.ndim == 2
        if single:
            history = history.unsqueeze(0)  # (1, L, k)
        w = self._torch_weights(history)
        rev = torch.flip(history, dims=[1])  # (N, L, k): lag L…1
        flat = rev.transpose(1, 2).reshape(history.shape[0], self.k * self.L)
        masked = flat.unsqueeze(1) * w["masked_weight"].unsqueeze(0)  # (N, k, k*L)
        if self.activation == "tanh":
            h = torch.tanh(torch.einsum("khd,nkd->nkh", w["W1"], masked) + w["b1"])
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
    def from_state_dict(cls, d: dict) -> "MLPMechanism":
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


def mechanism_from_state_dict(d: dict) -> Mechanism:
    """Dispatch on the ``"__type__"`` discriminator to the right subclass.

    Coerces the discriminator with ``str(...)`` because ``np.load`` returns it as
    a 0-d array.
    """
    mech_type = str(np.asarray(d["__type__"]).item()) if not isinstance(
        d["__type__"], str
    ) else d["__type__"]
    if mech_type == "linear":
        return LinearMechanism.from_state_dict(d)
    if mech_type == "mlp":
        return MLPMechanism.from_state_dict(d)
    raise ValueError(f"unknown mechanism __type__: {mech_type!r}")
