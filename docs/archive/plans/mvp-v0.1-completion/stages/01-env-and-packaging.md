# Stage 1: Environment & packaging fix

**Goal**: Establish a reproducible uv environment, fix the broken packaging so `pip install -e .` / install works, and get the existing test suite green as the baseline.
**Dependencies**: None

---

## Steps

1. Create the feature branch.
   - `git checkout -b mvp-v0.1`

2. Fix the invalid build backend.
   - File: `pyproject.toml`
   - Change `build-backend = "setuptools.backends.legacy:build"` → `build-backend = "setuptools.build_meta"`.
   - Keep `requires = ["setuptools>=68", "wheel"]`.

3. Initialise the uv environment and lockfile.
   - Run `uv sync --extra dev` (creates `.venv` + `uv.lock`, installs all `dependencies` + the `dev` extra).
   - Confirm imports: `uv run python -c "import torch, sklearn, scipy, numpy; print('env ok')"`.
   - Note: torch>=2.0 on Python — if 3.13 wheels are unavailable, pin `requires-python` and/or torch to a version with CPU wheels; document any pin in the index Issues section.

4. Reconcile dependency sources.
   - File: `requirements.txt`
   - Either delete it (uv is canonical) or add a one-line header comment pointing to `pyproject.toml` + `uv.lock` as the source of truth. Do **not** maintain two divergent lists. Recommended: keep a trimmed `requirements.txt` only if needed for non-uv consumers; otherwise remove and note in README.

5. Switch CI to uv.
   - File: `.github/workflows/ci.yml`
   - Replace `pip install -e ".[dev]"` + `pytest tests/` with the `astral-sh/setup-uv` action (or `pipx`-free curl install), then `uv sync --extra dev` and `uv run pytest tests/`.
   - Keep Python 3.11 (matches a stable torch CPU wheel) unless Step 3 forces a change.

6. Commit `uv.lock`.

7. **(Extension) Add the Pearl delta-recursion CF-faith metric alongside the existing one — keep both.**
   - File: `causaltemp_xai/metrics/cf_faith.py`
   - **Do not alter the committed noiseless-rollout behaviour (#1)** — it stays the **default** and the green baseline + `test_identical_cf_hard_is_zero` depend on it.
   - Add a `semantics` parameter: `CFfaith.__init__(self, tol=1e-4, scale=1.0, semantics="noiseless_rollout")`, accepting `"noiseless_rollout"` (#1, default) and `"pearl_delta"` (#3). Validate; raise `ValueError` on any other value (mirror the generator's `noise_type` validation style).
   - In `score()`, the **retroactive check is identical for both modes**. Branch only the forward simulation:
     - `noiseless_rollout` (#1): unchanged — re-simulate `x_cf` noiselessly from `intervention_t` and compare to `x_cf` (current `cf_faith.py:105-120`).
     - `pearl_delta` (#3): set `delta = x_cf − x_orig`; `sim = delta.copy()`; for `t > intervention_t`, `sim[t] = Σ_l A_l @ sim[t-l-1]`; `residual = mean|delta[t0:] − sim[t0:]|`. The original innovations cancel in `delta`, so this rewards CFs that differ from the factual **only by the propagated intervention** (on-manifold).
     - `hard`/`soft` are computed from `residual` by the same formulas in both modes.
   - **Why both (record in index Decisions):** the two metrics reward structurally different CFs — #1 rewards noiseless (off-manifold) rollouts; #3 rewards noise-reinjected (on-manifold) Pearl counterfactuals. A single CF **cannot** score `hard=1` on both, so reporting both is itself a benchmark result: the definition-sensitivity of "causal faithfulness" (strengthens the H4 disagreement story).
   - Tests: add a `TestCFFaithPearl` class to `tests/test_cf_faith.py`:
     - identical CF → `pearl` hard=1 (delta=0);
     - a **noise-reinjected** compliant CF (`x_cf[t] = Σ A·x_cf[t-l] + e[t]`, `e[t] = x_orig[t] − Σ A·x_orig[t-l]`) → `pearl` hard=1, **and** assert the same CF scores `noiseless_rollout` hard=0 (documents the divergence);
     - a **noiseless-built** CF (`x_cf[t] = Σ A·x_cf[t-l]`) → `pearl` hard=0, **and** `noiseless_rollout` hard=1 (mirror divergence);
     - a retroactive change → both modes hard=0.
   - **Downstream (forward-pointers — not implemented in this stage):** Stage 5 `eval.evaluate_method` computes **both** metrics for every CF; Stage 8 `results.json`/figures report both as `cf_faith_rollout_{hard,soft}` and `cf_faith_pearl_{hard,soft}`. CARLA (Stage 4) keeps emitting the noiseless rollout → `rollout` hard=1 / `pearl` hard=0 (an informative contrast, not a bug); an optional Pearl-CARLA variant (noise re-injection) is deferred.

---

## Verification

- [ ] `uv run python -c "import causaltemp_xai; print(causaltemp_xai.__version__)"` prints `0.1.0`.
- [ ] `uv run pytest tests/ -q` — all tests in `test_generator.py` and `test_cf_faith.py` pass (this is the green baseline).
- [ ] `uv.lock` exists and is committed.
- [ ] `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"` succeeds (valid toml) and build-backend reads `setuptools.build_meta`.
- [ ] (extension) `CFfaith()` default and `CFfaith(semantics="pearl_delta")` both import; an invalid `semantics` raises `ValueError`.
- [ ] (extension) `uv run pytest tests/test_cf_faith.py -q` passes including `TestCFFaithPearl`, with the both-mode divergence assertions (a CF that is `hard=1` under one mode is `hard=0` under the other).

---

## Commit

Two commits (the infra commit already landed as `592654a`; the metric extension is a follow-up):

1. `chore(infra): fix build backend, set up uv environment, green test baseline`
2. `feat(metrics): add Pearl delta-recursion CF-faith alongside noiseless-rollout (dual metric)`
