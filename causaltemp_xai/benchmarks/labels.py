"""Label functionals: which scalar of a trajectory the binary label thresholds.

**Why this is a benchmark axis and not a constant (RISK-19).** Every generator
here previously hardcoded ``Y = 1[x[-1, 0] > median]`` — a threshold on one
channel at the *terminal* timestep. That makes the label site identical to the
trajectory end, so the horizon claim (``general_plan.md`` §4.2, H8: an
intervention's effect on the label decays in ``T - t0``) and the weaker claim
"…decays in ``t_label - t0``" are **observationally identical**. H8 is not
falsifiable while only one label functional exists, and a reviewer can attribute
the whole horizon result to terminal labelling rather than to a property of
stable temporal SCMs.

Separating the two is what :data:`INTERIOR_THRESHOLD` exists for: it places the
label at ``0.6 T``, so ``t_label - t0`` and ``T - t0`` differ by a wide margin
and H8c (``docs/05_evaluation_plan.md``) can distinguish them.

Every functional here reduces a trajectory to one scalar and is thresholded at
its **median over the generated population**, so classes stay balanced by
construction and the label rule remains recoverable after the fact
(:func:`~causaltemp_xai.metrics.pns.recover_label_threshold` — ``theta`` is not
persisted, and the world-side PNS terms cannot be computed without it).

``TERMINAL_THRESHOLD`` is the default everywhere and reproduces the pre-2026-08-03
behaviour exactly; no committed preset or result changes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "INTERIOR_THRESHOLD",
    "TERMINAL_THRESHOLD",
    "LabelFunctional",
    "get_label_functional",
]


class LabelFunctional:
    """A trajectory → scalar reduction plus the label site it reads.

    Subclasses implement :meth:`latent_batch`; :meth:`latent_one` is derived
    from it so a functional cannot define the batch and single-instance paths
    inconsistently — which is exactly how a label rule silently stops matching
    its own recovered threshold.

    Attributes
    ----------
    name:
        Registry key, persisted in ``meta.json`` via ``BenchmarkConfig``.
    params:
        JSON-serialisable parameters, persisted alongside ``name``.
    """

    name: str = "abstract"

    def __init__(self, **params):
        self.params = dict(params)

    def latent_batch(self, X: np.ndarray) -> np.ndarray:
        """``(N, T, k)`` → ``(N,)`` — the scalar the threshold applies to."""
        raise NotImplementedError

    def latent_one(self, x: np.ndarray) -> float:
        """``(T, k)`` → scalar. Derived from :meth:`latent_batch`, never overridden."""
        return float(self.latent_batch(np.asarray(x, dtype=float)[None])[0])

    def label_site(self, T: int) -> int:
        """The last timestep this functional reads.

        Used to report the intervention-to-outcome distance ``t_label - t0``,
        which per ``general_plan.md`` §10 must accompany every validity and PNS
        number. For window functionals this is the *latest* timestep read — the
        optimistic end of the window, so the reported distance is never longer
        than the true one.
        """
        raise NotImplementedError

    def as_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.params})"


class TerminalThreshold(LabelFunctional):
    """``x[-1, channel]`` — the original rule. Label site is ``T - 1``."""

    name = "terminal_threshold"

    def __init__(self, channel: int = 0):
        super().__init__(channel=channel)
        self.channel = int(channel)

    def latent_batch(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, -1, self.channel]

    def label_site(self, T: int) -> int:
        return T - 1


class InteriorThreshold(LabelFunctional):
    """``x[t_label, channel]`` with ``t_label = int(frac * T)``.

    The point of the whole module: it moves the label off the trajectory end so
    that ``t_label - t0`` and ``T - t0`` are no longer the same number. The
    trajectory still runs to ``T``, so the *only* thing that changes relative to
    :class:`TerminalThreshold` is where the outcome is read — which is what
    makes it a controlled comparison rather than a shorter benchmark.
    """

    name = "interior_threshold"

    def __init__(self, frac: float = 0.6, channel: int = 0):
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1]; got {frac}")
        super().__init__(frac=frac, channel=channel)
        self.frac = float(frac)
        self.channel = int(channel)

    def _t_label(self, T: int) -> int:
        return min(int(self.frac * T), T - 1)

    def latent_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X[:, self._t_label(X.shape[1]), self.channel]

    def label_site(self, T: int) -> int:
        return self._t_label(T)


class WindowMean(LabelFunctional):
    """Mean of ``channel`` over the last ``window`` timesteps.

    A label that no single timestep determines, so a CF cannot flip it by
    rewriting one cell — the failure mode ``sparsity = 0.02`` methods exploit.
    """

    name = "window_mean"

    def __init__(self, window: int = 10, channel: int = 0):
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        super().__init__(window=window, channel=channel)
        self.window = int(window)
        self.channel = int(channel)

    def latent_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        w = min(self.window, X.shape[1])
        return X[:, -w:, self.channel].mean(axis=1)

    def label_site(self, T: int) -> int:
        return T - 1


class MultichannelLinear(LabelFunctional):
    """``w · x[-1, :]`` — a label depending on every channel at the endpoint.

    ``weights=None`` uses a uniform vector, resolved against ``k`` at call time
    so the functional does not need to know the config's width.
    """

    name = "multichannel_linear"

    def __init__(self, weights=None):
        super().__init__(weights=None if weights is None else list(map(float, weights)))
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def latent_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        k = X.shape[2]
        w = np.ones(k) / k if self.weights is None else self.weights
        if w.shape != (k,):
            raise ValueError(f"weights has shape {w.shape}, expected ({k},)")
        return X[:, -1, :] @ w

    def label_site(self, T: int) -> int:
        return T - 1


#: The default — bit-identical to the pre-2026-08-03 hardcoded rule.
TERMINAL_THRESHOLD = TerminalThreshold()

#: The H8c instrument: label at 0.6 T, trajectory still runs to T.
INTERIOR_THRESHOLD = InteriorThreshold(frac=0.6)

_REGISTRY = {
    TerminalThreshold.name: TerminalThreshold,
    InteriorThreshold.name: InteriorThreshold,
    WindowMean.name: WindowMean,
    MultichannelLinear.name: MultichannelLinear,
}


def get_label_functional(name: str | None = None, params: dict | None = None) -> LabelFunctional:
    """Resolve a label functional by registry name.

    ``None`` returns :data:`TERMINAL_THRESHOLD`, so every caller that does not
    opt in keeps the original rule.
    """
    if name is None:
        return TERMINAL_THRESHOLD
    if name not in _REGISTRY:
        raise ValueError(f"unknown label functional {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))
