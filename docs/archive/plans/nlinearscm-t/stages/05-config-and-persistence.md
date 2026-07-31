# Stage 5: Config presets + nonlinear persistence

**Goal**: Register NlinearSCM-T as first-class config presets and make `data_io` generate/persist/load nonlinear datasets via the `Mechanism.state_dict` contract.
**Dependencies**: Stages 1–3 (`Mechanism` serialization, `MLPMechanism`, `NlinearSCMT`).

---

## Steps

1. **Extend `BenchmarkConfig` in `causaltemp_xai/config.py`.**
   - Add a `mechanism_type: str = "linear"` field (values `"linear"` / `"mlp"`) and an optional `nonlinear: dict | None = None` (or explicit fields `hidden`, `gain`, `decay_min`, `decay_max`, `spectral_cap`, `init_gain`, `activation`) carrying nonlinear hyperparams. Keep the dataclass frozen and `as_dict()` JSON-serializable.
   - **Compatibility**: existing linear presets must serialize/behave identically (default `mechanism_type="linear"`, `nonlinear=None`). The Stage-1 golden test guards this.
   - Note the `L=1` invariant comment still applies (per-matrix spectral scaling / acyclic-across-time). NlinearSCM-T presets also use `L=1` unless a stage explicitly revisits stability for `L>1`.

2. **Add nonlinear presets.**
   - `SMOKE_NL` (CI/fast: small `k`, `T`, `N`, `hidden≈16`) and `FULL_NL` (paper-scale: mirror `FULL`'s `k=10, T=100, N=10_000` with nonlinear hyperparams).
   - Register both in `CONFIGS`. Confirm `shifted_config` still works for nonlinear (noise-only shift; the mechanism is seed-built before noise, so graph+mechanism stay bit-identical across the shift — same Axis-D invariant as linear; verify and document).

3. **Generalize `causaltemp_xai/data_io.py` to dispatch on `mechanism_type`.**
   - `generate_and_save`: build `NlinearSCMT` when `config.mechanism_type == "mlp"`, else `LinearSCMT`. Persist the mechanism via `mechanism.state_dict()` (the Stage-1 `mechanism.npz`); record `mechanism_type` and hyperparams in `meta.json`.
   - `load_dataset`: restore via `mechanism_from_state_dict` (Stage 1). Returns `mechanism` (object), `graph`, splits, `meta`.
   - **Output dir**: generalize `DEFAULT_OUT_DIR` from the hardcoded `data/linearscm_t` to a per-type/per-config layout (e.g. `data/<mechanism_type>scm_t/<config.name>/` or simply key by `config.name`). Update `.gitignore` so nonlinear data dirs are ignored too. Update the CLI `--config` choices (auto from `CONFIGS`).

4. **Update tests.**
   - Extend `tests/test_nlinear_generator.py` (or a new `tests/test_data_io_nlinear.py`) with a save→load round-trip for `SMOKE_NL`: `mechanism` round-trips (forward outputs bit-identical), both classes present per split, `meta.json` records `mechanism_type="mlp"` + hyperparams.
   - Ensure linear `tests/test_generator.py` persistence tests still pass against the generalized `data_io` (filename/dir changes reflected).

---

## Verification

> Run with `uv run --offline …` (see `resources/commands.md`).

- [ ] `uv run --offline python -m causaltemp_xai.data_io --config smoke_nl` writes a dataset; `load_dataset("smoke_nl")` round-trips.
- [ ] `uv run --offline pytest tests/test_nlinear_generator.py tests/test_generator.py -q` green (both linear & nonlinear persistence).
- [ ] `meta.json` records `mechanism_type` + nonlinear hyperparams; loaded mechanism reproduces forward outputs bit-for-bit.
- [ ] `shifted_config(SMOKE_NL)` yields a generator with bit-identical graph + mechanism (Axis-D invariant holds).
- [ ] Full suite green (`uv run --offline pytest -q`).

---

## Commit

`feat(config,data): register NlinearSCM-T presets + nonlinear dataset persistence`
