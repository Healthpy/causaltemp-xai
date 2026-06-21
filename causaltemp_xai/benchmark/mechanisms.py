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
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:  # torch is a hard dep elsewhere; import lazily-friendly for clarity
    import torch
except Exception:  # pragma: no cover - torch always present in this project
    torch = None  # type: ignore


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
    raise ValueError(f"unknown mechanism __type__: {mech_type!r}")
