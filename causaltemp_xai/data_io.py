"""Reproducible dataset generation, persistence, and loading for SCM-T.

Datasets are deterministic given a :class:`~causaltemp_xai.config.BenchmarkConfig`
(the SCM seed fixes the graph, mechanism, trajectories, and labels), so the
on-disk artifacts are **regenerated in CI** rather than committed — see the
plan's pinned decision and ``.gitignore`` (``data/`` is ignored).

Both mechanism families are handled transparently: ``config.mechanism_type``
selects :class:`~causaltemp_xai.benchmark.generator.LinearSCMT` (``"linear"``)
or :class:`~causaltemp_xai.benchmark.generator.NlinearSCMT` (``"mlp"``), and the
mechanism is persisted via its polymorphic ``state_dict`` / restored via
:func:`~causaltemp_xai.benchmark.mechanisms.mechanism_from_state_dict`. Each
config is keyed by its (unique) ``name`` on disk, so linear and nonlinear
datasets never collide.

Layout written per config (``out_dir/<config.name>/``)::

    X_train.npy  X_val.npy  X_test.npy        # (n_split, T, k)
    Y_train.npy  Y_val.npy  Y_test.npy        # (n_split,)
    graph.npy                                  # (k, k, L)
    mechanism.npz                              # Mechanism.state_dict() (+ __type__)
    meta.json                                  # config + per-split class balance

CLI::

    uv run python -m causaltemp_xai.data_io --config smoke
    uv run python -m causaltemp_xai.data_io --config smoke_nl
    uv run python -m causaltemp_xai.data_io --config full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causaltemp_xai.benchmarks.generator import (
    HMMRegimeSwitchNlinearSCMT,
    KuramotoSCMT,
    LinearSCMT,
    NlinearSCMT,
    RegimeSwitchNlinearSCMT,
    SpringSCMT,
)
from causaltemp_xai.benchmarks.mechanisms import mechanism_from_state_dict
from causaltemp_xai.config import (
    CONFIGS,
    BenchmarkConfig,
    get_config,
    shifted_config,
)

#: Default root directory for persisted datasets (gitignored). Mechanism-agnostic:
#: linear and nonlinear datasets coexist here, keyed by unique ``config.name``.
DEFAULT_OUT_DIR = Path("data/scm_t")

#: Train/val/test fractions.
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


def stratified_split(
    Y: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint train/val/test index arrays, stratified on ``Y``.

    Splitting is performed per class so every class appears in every split at
    (approximately) the requested fractions. The shuffle is seeded for
    reproducibility.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = round(n * fractions[0])
        n_val = round(n * fractions[1])
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])

    train = np.concatenate(train_idx)
    val = np.concatenate(val_idx)
    test = np.concatenate(test_idx)

    # Shuffle within each split so class blocks are interleaved.
    for split in (train, val, test):
        rng.shuffle(split)

    return train, val, test


def _class_balance(y: np.ndarray) -> dict[str, int]:
    """Return ``{class_label: count}`` as JSON-friendly string keys."""
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def build_generator(config: BenchmarkConfig):
    """Construct the SCM generator selected by ``config.mechanism_type``.

    ``"linear"`` → :class:`LinearSCMT`; ``"mlp"`` → :class:`NlinearSCMT` built
    with the ``config.nonlinear`` hyperparameters (``decay_range`` coerced from
    the JSON list back to a tuple); ``"mlp_regime_switch"`` →
    :class:`RegimeSwitchNlinearSCMT` (M4/H7 ablation) built with
    ``config.nonlinear``'s ``hidden``/``switch_frac``/``regime1``/``regime2``
    (each regime dict's ``decay_range`` coerced from list to tuple). Raises
    ``ValueError`` on an unknown type.
    """
    if config.mechanism_type == "linear":
        return LinearSCMT(
            k=config.k,
            L=config.L,
            sparsity=config.sparsity,
            noise_type=config.noise_type,
            T=config.T,
            N=config.N,
            seed=config.seed,
            label_fn=config.label_fn,
            label_params=config.label_params,
        )
    if config.mechanism_type == "mlp":
        nl = dict(config.nonlinear or {})
        if "decay_range" in nl:
            nl["decay_range"] = tuple(nl["decay_range"])
        return NlinearSCMT(
            k=config.k,
            L=config.L,
            sparsity=config.sparsity,
            noise_type=config.noise_type,
            T=config.T,
            N=config.N,
            seed=config.seed,
            label_fn=config.label_fn,
            label_params=config.label_params,
            **nl,
        )
    if config.mechanism_type == "mlp_regime_switch":
        nl = dict(config.nonlinear or {})
        regime1 = dict(nl.get("regime1", {}))
        regime2 = dict(nl.get("regime2", {}))
        if "decay_range" in regime1:
            regime1["decay_range"] = tuple(regime1["decay_range"])
        if "decay_range" in regime2:
            regime2["decay_range"] = tuple(regime2["decay_range"])
        return RegimeSwitchNlinearSCMT(
            k=config.k,
            L=config.L,
            sparsity=config.sparsity,
            noise_type=config.noise_type,
            T=config.T,
            N=config.N,
            seed=config.seed,
            label_fn=config.label_fn,
            label_params=config.label_params,
            hidden=nl.get("hidden", 16),
            switch_frac=nl.get("switch_frac", 0.5),
            regime1=regime1 or None,
            regime2=regime2 or None,
        )
    if config.mechanism_type == "mlp_regime_hmm":
        nl = dict(config.nonlinear or {})
        regimes = nl.get("regimes")
        if regimes is not None:
            regimes = [
                {**r, "decay_range": tuple(r["decay_range"])} if "decay_range" in r else dict(r)
                for r in regimes
            ]
        return HMMRegimeSwitchNlinearSCMT(
            k=config.k,
            L=config.L,
            sparsity=config.sparsity,
            noise_type=config.noise_type,
            T=config.T,
            N=config.N,
            seed=config.seed,
            label_fn=config.label_fn,
            label_params=config.label_params,
            hidden=nl.get("hidden", 16),
            n_regimes=nl.get("n_regimes", 3),
            p_stay=nl.get("p_stay", 0.9),
            regimes=regimes,
        )
    if config.mechanism_type == "spring":
        nl = dict(config.nonlinear or {})
        # config.k is the exposed channel count (2 * n_particles), matching
        # every other family's "config.k == X.shape[-1]" convention --
        # SpringSCMT itself takes n_particles, so it's derived here.
        if config.k % 2 != 0:
            raise ValueError(f"'spring' mechanism_type requires an even config.k, got {config.k}")
        return SpringSCMT(
            n_particles=config.k // 2,
            sparsity=config.sparsity,
            noise_type=config.noise_type,
            T=config.T,
            N=config.N,
            seed=config.seed,
            label_fn=config.label_fn,
            label_params=config.label_params,
            **nl,
        )
    if config.mechanism_type == "kuramoto":
        nl = dict(config.nonlinear or {})
        if "omega_range" in nl:
            nl["omega_range"] = tuple(nl["omega_range"])
        return KuramotoSCMT(
            k=config.k,
            sparsity=config.sparsity,
            noise_type=config.noise_type,
            T=config.T,
            N=config.N,
            seed=config.seed,
            label_fn=config.label_fn,
            label_params=config.label_params,
            **nl,
        )
    raise ValueError(
        f"unknown mechanism_type {config.mechanism_type!r}; "
        "expected 'linear', 'mlp', 'mlp_regime_switch', 'mlp_regime_hmm', "
        "'spring', or 'kuramoto'"
    )


def generate_and_save(
    config: BenchmarkConfig,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> Path:
    """Generate a dataset from ``config`` and persist it to disk.

    Dispatches on ``config.mechanism_type`` (linear VAR vs nonlinear MLP) via
    :func:`build_generator`. Returns the directory the artifacts were written to
    (``out_dir/<config.name>/``).
    """
    out_dir = Path(out_dir)
    dest = out_dir / config.name
    dest.mkdir(parents=True, exist_ok=True)

    gen = build_generator(config)
    data = gen.generate()
    X, Y = data["X"], data["Y"]
    graph, mechanism = data["graph"], data["mechanism"]

    train_idx, val_idx, test_idx = stratified_split(Y, seed=config.seed)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split_name, idx in splits.items():
        np.save(dest / f"X_{split_name}.npy", X[idx])
        np.save(dest / f"Y_{split_name}.npy", Y[idx])

    np.save(dest / "graph.npy", graph)
    np.savez(dest / "mechanism.npz", **mechanism.state_dict())

    # M4/H7 regime-switch ablation: persist the second regime's mechanism too,
    # if present (absent for every other mechanism_type -- zero behavior
    # change for existing "linear"/"mlp" presets).
    mechanism_regime2 = data.get("mechanism_regime2")
    if mechanism_regime2 is not None:
        np.savez(dest / "mechanism_regime2.npz", **mechanism_regime2.state_dict())

    # M4/H7 HMM regime-switch ablation ("mlp_regime_hmm"): persist every
    # regime's mechanism (regime 0 is already saved as mechanism.npz above),
    # the transition matrix, and the per-sequence ground-truth regime path.
    # Absent for every other mechanism_type -- zero change for existing presets.
    regime_mechanisms = data.get("mechanisms")
    if regime_mechanisms is not None and len(regime_mechanisms) > 1:
        for r, mech in enumerate(regime_mechanisms):
            np.savez(dest / f"mechanism_regime{r}.npz", **mech.state_dict())
        np.save(dest / "transition_matrix.npy", data["transition_matrix"])
        # regime_path is (N, T); split it to match the X_{split}.npy layout so
        # a loaded split's regime path aligns row-for-row with its X/Y.
        for split_name, idx in splits.items():
            np.save(dest / f"regime_path_{split_name}.npy", data["regime_path"][idx])

    meta = {
        "config": config.as_dict(),
        "mechanism_type": str(mechanism.state_dict()["__type__"]),
        "nonlinear": config.nonlinear,
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
        "class_balance": {name: _class_balance(Y[idx]) for name, idx in splits.items()},
        "shapes": {
            "X_train": list(X[train_idx].shape),
            "graph": list(graph.shape),
        },
    }
    if "switch_t" in data:
        meta["switch_t"] = int(data["switch_t"])
    with open(dest / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return dest


def load_dataset(
    config_name: str,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> dict:
    """Load a previously persisted dataset.

    Returns a dict with keys ``X_train/X_val/X_test``, ``Y_train/Y_val/Y_test``,
    ``graph``, ``mechanism`` (a :class:`~causaltemp_xai.benchmark.mechanisms.Mechanism`),
    and ``meta``. For the M4/H7 regime-switch preset (``mechanism_type ==
    "mlp_regime_switch"``) also includes ``mechanism_regime2`` (present only
    if ``mechanism_regime2.npz`` exists on disk -- absent for every other
    preset); ``meta["switch_t"]`` carries the regime-switch index into the
    observed ``(T,)`` window.
    """
    src = Path(out_dir) / config_name
    if not src.exists():
        raise FileNotFoundError(
            f"no dataset at {src}; run `python -m causaltemp_xai.data_io "
            f"--config {config_name}` first"
        )

    out: dict = {}
    for split_name in ("train", "val", "test"):
        out[f"X_{split_name}"] = np.load(src / f"X_{split_name}.npy")
        out[f"Y_{split_name}"] = np.load(src / f"Y_{split_name}.npy")

    out["graph"] = np.load(src / "graph.npy")

    with np.load(src / "mechanism.npz") as mech:
        out["mechanism"] = mechanism_from_state_dict(dict(mech))

    # M4/H7 regime-switch ablation: optional second-regime mechanism.
    regime2_path = src / "mechanism_regime2.npz"
    if regime2_path.exists():
        with np.load(regime2_path) as mech2:
            out["mechanism_regime2"] = mechanism_from_state_dict(dict(mech2))

    # M4/H7 HMM regime-switch ablation ("mlp_regime_hmm"): optional full
    # regime set + transition matrix + per-split regime paths. Present only if
    # transition_matrix.npy was written (absent for every other preset).
    tm_path = src / "transition_matrix.npy"
    if tm_path.exists():
        out["transition_matrix"] = np.load(tm_path)
        mechanisms = []
        r = 0
        while (src / f"mechanism_regime{r}.npz").exists():
            with np.load(src / f"mechanism_regime{r}.npz") as mech_r:
                mechanisms.append(mechanism_from_state_dict(dict(mech_r)))
            r += 1
        out["mechanisms"] = mechanisms
        for split_name in ("train", "val", "test"):
            rp_path = src / f"regime_path_{split_name}.npy"
            if rp_path.exists():
                out[f"regime_path_{split_name}"] = np.load(rp_path)

    with open(src / "meta.json") as fh:
        out["meta"] = json.load(fh)

    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and persist an SCM-T dataset (linear or nonlinear)."
    )
    parser.add_argument(
        "--config",
        choices=sorted(CONFIGS),
        required=True,
        help="Named benchmark config to generate.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output root directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--shift-noise",
        choices=("laplace", "uniform"),
        default=None,
        help=(
            "Generate the Shift-VR environment instead: same SCM (graph/"
            "mechanisms) as --config but this innovation distribution. Saved "
            "under '<config>_shift'."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    config = get_config(args.config)
    if args.shift_noise is not None:
        config = shifted_config(config, noise_type=args.shift_noise)
    dest = generate_and_save(config, out_dir=args.out_dir)

    meta_path = dest / "meta.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    print(f"Wrote dataset for config '{config.name}' to {dest}")
    print(f"  split sizes:   {meta['split_sizes']}")
    print(f"  class balance: {meta['class_balance']}")


if __name__ == "__main__":
    main()
