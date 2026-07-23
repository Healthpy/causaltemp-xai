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

Additionally, three M4 benchmark-extension **ablation presets** are
registered below, at the same smoke scale as ``SMOKE``/``SMOKE_NL`` (k=5,
T=30, N=500, seed=0) -- ``SMOKE_GAUSSIAN`` (H5 negative control),
``SMOKE_NONMONOTONIC`` (H6 non-monotonic mechanism), and ``SMOKE_REGIME`` (H7
regime-switching). See ``docs/m4_ablation_presets_smoke.md`` for the full
design, pre-registered expected direction, and smoke-scale preliminary
finding for each. **No full-scale variant of any of these three presets
exists or is planned as part of this work.**
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


# ---------------------------------------------------------------------------
# M4 benchmark-extension ablation presets (H5/H6/H7) -- SMOKE-SCALE ONLY
# ---------------------------------------------------------------------------
#
# Each preset below varies exactly one field/hyperparameter-group from a
# locked SMOKE-family preset, mirroring shifted_config/seeded_variant's
# "vary exactly one field" convention -- but, like SMOKE_NL, each is
# registered as a first-class named preset (not built ad hoc per
# invocation) so it can be passed directly as `--config <name>` to the
# phased experiment pipeline (experiments/01-04). All three are built at
# the same smoke scale as SMOKE/SMOKE_NL (k=5, T=30, N=500, seed=0).
#
# See docs/m4_ablation_presets_smoke.md for the full design + pre-registered
# expected direction + smoke-scale preliminary finding for each. **No
# full-scale ("full"/"full_nl"-analogue) variant of any of these three
# presets exists or is planned as part of this work.**

#: H5 negative control: SMOKE with Gaussian (instead of Laplace) innovation
#: noise, variance-matched to Laplace(scale=0.1) so the ablation isolates
#: noise *shape*, not innovation scale (see generator._GAUSSIAN_STD).
#: Pre-registered expected direction: weakens or nulls the validity/CF-faith
#: divergence that is this project's core phenomenon (H5).
SMOKE_GAUSSIAN = BenchmarkConfig(
    k=SMOKE.k,
    L=SMOKE.L,
    sparsity=SMOKE.sparsity,
    noise_type="gaussian",
    T=SMOKE.T,
    N=SMOKE.N,
    seed=SMOKE.seed,
    name="smoke_gaussian",
    mechanism_type=SMOKE.mechanism_type,
    nonlinear=dict(SMOKE.nonlinear) if SMOKE.nonlinear is not None else None,
)

#: H6 non-monotonic mechanism ablation: SMOKE_NL with the per-node MLP's
#: hidden activation swapped from "tanh" (monotonic) to "nonmonotonic" (a
#: bounded, odd `sin` activation -- see mechanisms._ACTIVATIONS_NP). The
#: mechanism's **output** branch stays `gain*tanh(...)` regardless (see the
#: MLPMechanism docstring), so the global boundedness guarantee is
#: unaffected -- only the hidden representation's monotonicity varies.
_NL_HYPERPARAMS_NONMONOTONIC: dict = {**_NL_HYPERPARAMS, "activation": "nonmonotonic"}

SMOKE_NONMONOTONIC = BenchmarkConfig(
    k=SMOKE_NL.k,
    L=SMOKE_NL.L,
    sparsity=SMOKE_NL.sparsity,
    noise_type=SMOKE_NL.noise_type,
    T=SMOKE_NL.T,
    N=SMOKE_NL.N,
    seed=SMOKE_NL.seed,
    name="smoke_nonmonotonic",
    mechanism_type="mlp",
    nonlinear=dict(_NL_HYPERPARAMS_NONMONOTONIC),
)

#: H7 regime-switching ablation: SMOKE_NL-scale dataset with a single
#: deterministic structural break at T/2
#: (see ``causaltemp_xai.benchmarks.generator.RegimeSwitchNlinearSCMT``).
#: Regime 1 is identical to `_NL_HYPERPARAMS` (the standard, un-ablated
#: nonlinear mechanism family -- same as SMOKE_NL) so the ablation isolates
#: exactly one thing: the presence of a second, post-switch regime. Regime 2
#: uses a non-overlapping `decay_range` + distinct `gain`/`spectral_cap`
#: (deterministically, not just statistically, different effective
#: parameters). `mechanism_type="mlp_regime_switch"` is dispatched by
#: `data_io.build_generator`.
_REGIME_NL_HYPERPARAMS: dict = {
    "hidden": _NL_HYPERPARAMS["hidden"],
    "switch_frac": 0.5,
    "regime1": {
        "decay_range": list(_NL_HYPERPARAMS["decay_range"]),
        "gain": _NL_HYPERPARAMS["gain"],
        "spectral_cap": _NL_HYPERPARAMS["spectral_cap"],
        "init_gain": _NL_HYPERPARAMS["init_gain"],
        "activation": _NL_HYPERPARAMS["activation"],
    },
    "regime2": {
        "decay_range": [0.05, 0.25],
        "gain": 0.35,
        "spectral_cap": 0.5,
        "init_gain": 0.35,
        "activation": "tanh",
    },
}

SMOKE_REGIME = BenchmarkConfig(
    k=SMOKE_NL.k,
    L=SMOKE_NL.L,
    sparsity=SMOKE_NL.sparsity,
    noise_type=SMOKE_NL.noise_type,
    T=SMOKE_NL.T,
    N=SMOKE_NL.N,
    seed=SMOKE_NL.seed,
    name="smoke_regime",
    mechanism_type="mlp_regime_switch",
    nonlinear=dict(_REGIME_NL_HYPERPARAMS),
)

#: H7 regime-switching ablation, **HMM variant**: SMOKE_NL-scale dataset with
#: a genuine hidden Markov regime path (``R=3`` regimes, per-sequence random
#: change-points) rather than SMOKE_REGIME's single deterministic T/2 break
#: (see ``causaltemp_xai.benchmarks.generator.HMMRegimeSwitchNlinearSCMT``).
#: Regime 0 is identical to `_NL_HYPERPARAMS` (the standard, un-ablated
#: nonlinear mechanism family -- same as SMOKE_NL), so the ablation isolates
#: exactly the HMM structure. `mechanism_type="mlp_regime_hmm"` is dispatched
#: by `data_io.build_generator`. Kept alongside `smoke_regime` (not replacing
#: it) so the deterministic-single-break vs. stochastic-multi-break contrast
#: is a clean A/B for the H7 writeup.
_REGIME_HMM_HYPERPARAMS: dict = {
    "hidden": _NL_HYPERPARAMS["hidden"],
    "n_regimes": 3,
    "p_stay": 0.9,
    "regimes": [
        {
            "decay_range": list(_NL_HYPERPARAMS["decay_range"]),
            "gain": _NL_HYPERPARAMS["gain"],
            "spectral_cap": _NL_HYPERPARAMS["spectral_cap"],
            "init_gain": _NL_HYPERPARAMS["init_gain"],
            "activation": _NL_HYPERPARAMS["activation"],
        },
        {
            "decay_range": [0.05, 0.25],
            "gain": 0.35,
            "spectral_cap": 0.5,
            "init_gain": 0.35,
            "activation": "tanh",
        },
        {
            "decay_range": [0.15, 0.28],
            "gain": 0.55,
            "spectral_cap": 0.7,
            "init_gain": 0.5,
            "activation": "tanh",
        },
    ],
}

SMOKE_REGIME_HMM = BenchmarkConfig(
    k=SMOKE_NL.k,
    L=SMOKE_NL.L,
    sparsity=SMOKE_NL.sparsity,
    noise_type=SMOKE_NL.noise_type,
    T=SMOKE_NL.T,
    N=SMOKE_NL.N,
    seed=SMOKE_NL.seed,
    name="smoke_regime_hmm",
    mechanism_type="mlp_regime_hmm",
    nonlinear=dict(_REGIME_HMM_HYPERPARAMS),
)


#: Registry of all named configs.
CONFIGS: dict[str, BenchmarkConfig] = {
    "smoke": SMOKE,
    "full": FULL,
    "full_sparse": FULL_SPARSE,
    "smoke_nl": SMOKE_NL,
    "full_nl": FULL_NL,
    "smoke_gaussian": SMOKE_GAUSSIAN,
    "smoke_nonmonotonic": SMOKE_NONMONOTONIC,
    "smoke_regime": SMOKE_REGIME,
    "smoke_regime_hmm": SMOKE_REGIME_HMM,
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
        raise ValueError(f"noise_type must be 'laplace' or 'uniform', got {noise_type!r}")
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


def seeded_variant(
    base: BenchmarkConfig,
    seed: int,
    name: str | None = None,
) -> BenchmarkConfig:
    """Derive a multi-seed replicate of ``base`` by changing only ``seed`` (M2, O2).

    Returns a new config identical to ``base`` (same ``k``, ``L``, ``sparsity``,
    ``noise_type``, ``T``, ``N``, ``mechanism_type``, ``nonlinear``) except for
    ``seed`` and ``name`` -- mirrors :func:`shifted_config`'s "vary exactly one
    field" pattern, but for the multi-seed protocol (plan Standing Decision #4:
    every headline number carries a CI from M2 onward, which requires >=5 seeds
    through generator -> classifier -> CF selection). Because the SCM seed
    drives the graph, the mechanism weights, and the trajectories in one shot
    (:func:`causaltemp_xai.data_io.build_generator`), and the dataset split
    (:func:`causaltemp_xai.data_io.stratified_split`) and the classifier
    (``LSTMClassifier(seed=cfg.seed, ...)``) are both seeded from
    ``cfg.seed`` too, changing only this one field re-seeds *every* stage of
    the pipeline in one call -- generation, split, training, and (since CF
    selection in ``experiments/_common.select_flip_candidates`` is a
    deterministic function of the seeded ``X_test`` order) CF-instance
    selection. Not registered in :data:`CONFIGS`: like ``shifted_config``,
    this is built ad hoc per invocation, keyed by the returned config's
    (unique) ``name`` on disk.

    Parameters
    ----------
    base:
        The registered preset to replicate (e.g. ``SMOKE``, ``FULL``).
    seed:
        The new seed. Deliberately **not** special-cased when it equals
        ``base.seed`` -- the returned config still gets a distinct ``name``
        (default ``f"{base.name}_seed{seed}"``) and therefore a fresh on-disk
        directory, so every seed replicate (including one that happens to
        reuse the base preset's own seed value) is unambiguous and never
        silently aliases the original un-suffixed run.
    name:
        Override the default ``f"{base.name}_seed{seed}"`` name.
    """
    return BenchmarkConfig(
        k=base.k,
        L=base.L,
        sparsity=base.sparsity,
        noise_type=base.noise_type,
        T=base.T,
        N=base.N,
        seed=seed,
        name=name or f"{base.name}_seed{seed}",
        mechanism_type=base.mechanism_type,
        nonlinear=dict(base.nonlinear) if base.nonlinear is not None else None,
    )


def get_config(name: str) -> BenchmarkConfig:
    """Look up a preset by name.

    Parameters
    ----------
    name:
        One of ``"smoke"``, ``"full"``, ``"full_sparse"``, ``"smoke_nl"``,
        ``"full_nl"``, ``"smoke_gaussian"``, ``"smoke_nonmonotonic"``,
        ``"smoke_regime"``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered config.
    """
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; choose one of {sorted(CONFIGS)}")
    return CONFIGS[name]
