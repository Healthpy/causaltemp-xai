# Stage 2: Benchmark config & dataset persistence

**Goal**: Lock the canonical LinearSCM-T configuration (two-tier smoke/full), add a config object and a reproducible data-generation CLI that persists X/Y/graph/mechanisms with 60/20/20 splits, and extend generator tests.
**Dependencies**: Stage 1

---

## Steps

1. Add a config module.
   - File: `causaltemp_xai/config.py` (new)
   - Define a frozen `@dataclass BenchmarkConfig` with fields `k, L, sparsity, noise_type, T, N, seed, name`.
   - Provide module-level presets `SMOKE` and `FULL` exactly as in [resources/configs.md](../resources/configs.md). Add a `CONFIGS = {"smoke": SMOKE, "full": FULL}` registry and a `get_config(name)` helper.
   - (Optional) add `FULL_SPARSE` (sparsity=0.1) for the H5 hint.

2. Add a data I/O module + CLI.
   - File: `causaltemp_xai/data_io.py` (new)
   - `generate_and_save(config: BenchmarkConfig, out_dir)`:
     - instantiate `LinearSCMT(k, L, sparsity, noise_type, T, N, seed)`, call `.generate()`.
     - split N into 60/20/20 train/val/test (seeded shuffle; stratify on Y so both classes appear in each split).
     - save to `data/linearscm_t/<config.name>/`: `X_train.npy, X_val.npy, X_test.npy, Y_train.npy, Y_val.npy, Y_test.npy`, plus `graph.npy` and `mechanisms.npz` (the L coefficient matrices), and a `meta.json` recording the config + per-split class balance.
   - `load_dataset(config_name, out_dir) -> dict` mirror.
   - `__main__`: `argparse` with `--config {smoke,full,full_sparse}` and `--out-dir` (default `data/linearscm_t`).
   - **Decision (pinned):** keep `data/linearscm_t/` gitignored for **all** configs and **regenerate in CI** — generation is deterministic via `seed`, so committing binary `.npy`/`.npz` artifacts only adds churn. Confirm `.gitignore` covers the directory; do not commit datasets.

3. Extend generator tests.
   - File: `tests/test_generator.py`
   - Add: (a) acyclicity check — for the locked configs assert there are no lag-0/contemporaneous edges (VAR is acyclic across time by construction; see configs.md note); (b) split integrity — train/val/test sizes are 60/20/20 of N and disjoint; (c) both classes present in every split; (d) round-trip `generate_and_save` → `load_dataset` reproduces arrays (`np.array_equal`).

---

## Verification

- [ ] `uv run python -m causaltemp_xai.data_io --config smoke` writes all 6 split arrays + `graph.npy` + `mechanisms.npz` + `meta.json` under `data/linearscm_t/smoke/`.
- [ ] `uv run pytest tests/test_generator.py -q` passes including the new acyclicity, split, and round-trip tests.
- [ ] `meta.json` shows ≈60/20/20 split and both classes in each split.

---

## Commit

`feat(benchmark): lock smoke/full configs and add reproducible dataset persistence`
