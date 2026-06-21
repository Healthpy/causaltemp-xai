"""Benchmark configuration presets for CausalTemp-XAI.

Two canonical tiers are locked here (see ``docs/plans/mvp-v0.1-completion/
resources/configs.md``):

* ``SMOKE`` — small/fast config used by tests and CI.
* ``FULL`` — the locked paper configuration, run once for results.

``FULL_SPARSE`` is an optional descriptive-only variant (H5 hint, no ablation
claim). All presets fix ``L=1``: with a single lag the VAR is trivially acyclic
across time and ``generator._stabilise`` guarantees stationarity. Do not bump
``L`` without revisiting ``_stabilise`` (per-matrix spectral scaling is only
valid at ``L=1``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable specification of an SCM-T benchmark instance.

    Field order mirrors the ``LinearSCMT`` / ``NlinearSCMT`` constructor (``k, L,
    sparsity, noise_type, T, N, seed``) with a human-readable ``name`` for on-disk
    layout and provenance.

    ``mechanism_type`` selects the transition mechanism family — ``"linear"``
    (VAR, :class:`~causaltemp_xai.benchmark.generator.LinearSCMT`) or ``"mlp"``
    (additive-noise per-node MLP,
    :class:`~causaltemp_xai.benchmark.generator.NlinearSCMT`). ``nonlinear``
    carries the MLP hyperparameters (``hidden``, ``gain``, ``decay_range``,
    ``spectral_cap``, ``init_gain``, ``activation``) and is ``None`` for the
    linear family. The defaults (``"linear"`` / ``None``) keep every existing
    preset byte-identical — the Stage-1 golden test guards this.
    """

    k: int
    L: int
    sparsity: float
    noise_type: str
    T: int
    N: int
    seed: int
    name: str
    mechanism_type: str = "linear"
    nonlinear: dict | None = None

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict of the config (for ``meta.json``)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Locked presets (see resources/configs.md)
# ---------------------------------------------------------------------------

#: CI / fast iteration / tests.
SMOKE = BenchmarkConfig(
    k=5,
    L=1,
    sparsity=0.2,
    noise_type="laplace",
    T=30,
    N=500,
    seed=0,
    name="smoke",
)

#: Locked paper configuration (MVP §WP1), run once for results.
FULL = BenchmarkConfig(
    k=10,
    L=1,
    sparsity=0.2,
    noise_type="laplace",
    T=100,
    N=10_000,
    seed=42,
    name="full",
)

#: Optional H5-hint variant: FULL but sparser. Descriptive only — no ablation claim.
FULL_SPARSE = BenchmarkConfig(
    k=10,
    L=1,
    sparsity=0.1,
    noise_type="laplace",
    T=100,
    N=10_000,
    seed=42,
    name="full_sparse",
)


# ---------------------------------------------------------------------------
# Nonlinear presets (NlinearSCM-T — additive-noise per-node MLP transitions)
# ---------------------------------------------------------------------------
#
# Both fix ``L=1`` like the linear tiers: the per-node MLP is contractive (leaky
# decay + bounded ``gain·tanh`` with spectral-norm-capped weights) so a single lag
# keeps the dynamics acyclic-across-time and bounded. Do not bump ``L`` without
# revisiting NlinearSCMT's stability recipe. ``decay_range`` is stored as a list
# so ``as_dict()`` stays JSON-serialisable; the generator coerces it to a tuple.

#: Default nonlinear MLP hyperparameters (mirrors NlinearSCMT's defaults).
_NL_HYPERPARAMS: dict = {
    "hidden": 16,
    "gain": 0.8,
    "decay_range": [0.3, 0.8],
    "spectral_cap": 0.9,
    "init_gain": 0.7,
    "activation": "tanh",
}

#: CI / fast iteration / tests — nonlinear analogue of ``SMOKE``.
SMOKE_NL = BenchmarkConfig(
    k=5,
    L=1,
    sparsity=0.2,
    noise_type="laplace",
    T=30,
    N=500,
    seed=0,
    name="smoke_nl",
    mechanism_type="mlp",
    nonlinear=dict(_NL_HYPERPARAMS),
)

#: Paper-scale nonlinear configuration — mirrors ``FULL`` with MLP mechanisms.
FULL_NL = BenchmarkConfig(
    k=10,
    L=1,
    sparsity=0.2,
    noise_type="laplace",
    T=100,
    N=10_000,
    seed=42,
    name="full_nl",
    mechanism_type="mlp",
    nonlinear=dict(_NL_HYPERPARAMS),
)


#: Registry of all named configs.
CONFIGS: dict[str, BenchmarkConfig] = {
    "smoke": SMOKE,
    "full": FULL,
    "full_sparse": FULL_SPARSE,
    "smoke_nl": SMOKE_NL,
    "full_nl": FULL_NL,
}


def shifted_config(
    base: BenchmarkConfig,
    noise_type: str = "uniform",
    name: str | None = None,
) -> BenchmarkConfig:
    """Derive a Shift-VR environment from ``base`` by changing only the noise.

    Returns a new config identical to ``base`` (same ``k``, ``L``, ``sparsity``,
    ``seed``, ``mechanism_type`` and ``nonlinear`` hyperparams) except for
    ``noise_type``. Because both :class:`LinearSCMT` and :class:`NlinearSCMT`
    build the graph and mechanism in ``__init__`` from the seed *before* and
    independent of the noise distribution, a generator built from this config has
    a **bit-identical** ``graph`` and ``mechanism`` to one built from ``base`` —
    isolating a pure innovation-distribution shift (Axis D), per the pinned
    Shift-VR protocol in ``resources/configs.md``. This holds for the nonlinear
    family too (the MLP weights are seed-built before any noise is drawn).
    """
    if noise_type not in ("laplace", "uniform"):
        raise ValueError(
            f"noise_type must be 'laplace' or 'uniform', got {noise_type!r}"
        )
    return BenchmarkConfig(
        k=base.k,
        L=base.L,
        sparsity=base.sparsity,
        noise_type=noise_type,
        T=base.T,
        N=base.N,
        seed=base.seed,
        name=name or f"{base.name}_shift",
        mechanism_type=base.mechanism_type,
        nonlinear=dict(base.nonlinear) if base.nonlinear is not None else None,
    )


def get_config(name: str) -> BenchmarkConfig:
    """Look up a preset by name.

    Parameters
    ----------
    name:
        One of ``"smoke"``, ``"full"``, ``"full_sparse"``, ``"smoke_nl"``,
        ``"full_nl"``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered config.
    """
    if name not in CONFIGS:
        raise KeyError(
            f"unknown config {name!r}; choose one of {sorted(CONFIGS)}"
        )
    return CONFIGS[name]
