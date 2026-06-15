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
    """Immutable specification of a LinearSCM-T benchmark instance.

    Field order mirrors the ``LinearSCMT`` constructor (``k, L, sparsity,
    noise_type, T, N, seed``) with a human-readable ``name`` for on-disk
    layout and provenance.
    """

    k: int
    L: int
    sparsity: float
    noise_type: str
    T: int
    N: int
    seed: int
    name: str

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


#: Registry of all named configs.
CONFIGS: dict[str, BenchmarkConfig] = {
    "smoke": SMOKE,
    "full": FULL,
    "full_sparse": FULL_SPARSE,
}


def shifted_config(
    base: BenchmarkConfig,
    noise_type: str = "uniform",
    name: str | None = None,
) -> BenchmarkConfig:
    """Derive a Shift-VR environment from ``base`` by changing only the noise.

    Returns a new config identical to ``base`` (same ``k``, ``L``, ``sparsity``,
    ``seed``) except for ``noise_type``. Because :class:`LinearSCMT` builds the
    graph and mechanisms in ``__init__`` from the seed *before* and independent
    of the noise distribution, a generator built from this config has a
    **bit-identical** ``graph`` and ``mechanisms`` to one built from ``base`` —
    isolating a pure innovation-distribution shift (Axis D), per the pinned
    Shift-VR protocol in ``resources/configs.md``.
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
    )


def get_config(name: str) -> BenchmarkConfig:
    """Look up a preset by name.

    Parameters
    ----------
    name:
        One of ``"smoke"``, ``"full"``, ``"full_sparse"``.

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
