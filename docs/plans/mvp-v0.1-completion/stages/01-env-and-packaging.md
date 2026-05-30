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

---

## Verification

- [ ] `uv run python -c "import causaltemp_xai; print(causaltemp_xai.__version__)"` prints `0.1.0`.
- [ ] `uv run pytest tests/ -q` — all tests in `test_generator.py` and `test_cf_faith.py` pass (this is the green baseline).
- [ ] `uv.lock` exists and is committed.
- [ ] `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"` succeeds (valid toml) and build-backend reads `setuptools.build_meta`.

---

## Commit

`chore(infra): fix build backend, set up uv environment, green test baseline`
