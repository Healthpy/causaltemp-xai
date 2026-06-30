# Commands

All commands use `uv` per team convention. Run from repo root.

> ⚠️ **Always pass `--offline`.** Bare `uv run …` re-resolves the environment against a
> private GitLab index (`gitlab.com/.../packages/pypi/simple`) that returns **401
> Unauthorized** for `setuptools>=68`, so it dies with "unsatisfiable requirements"
> *before running anything*. The local `.venv` is already complete, so `uv run --offline …`
> skips the resolve and works. Equivalent fallback: `.venv/bin/python -m pytest …`.

## Test

```bash
uv run --offline pytest -q                              # full suite (must stay green every stage)
uv run --offline pytest tests/test_golden_linear.py -v  # Stage 1: pinned v0.1 behavior (must not move)
uv run --offline pytest tests/test_mechanisms.py -v     # Stage 2
uv run --offline pytest tests/test_nlinear_generator.py -v   # Stage 3
uv run --offline pytest tests/test_structural_cf.py tests/test_cf_faith.py -v  # Stage 4
```

## Generate datasets

```bash
uv run --offline python -m causaltemp_xai.data_io --config smoke      # linear (existing)
uv run --offline python -m causaltemp_xai.data_io --config smoke_nl   # nonlinear (Stage 5)
uv run --offline python -m causaltemp_xai.data_io --config full_nl    # paper scale (Stage 5)
```

## Run the experiment harness

```bash
uv run --offline python experiments/run_all.py --config smoke      # linear smoke
uv run --offline python experiments/run_all.py --config smoke_nl   # nonlinear smoke (Stage 6, oracle CF)
```

## Branch / PR workflow (per 2026-06-18 sync)

```bash
git checkout -b feat/nlinearscm-t       # branch off main — do NOT commit on main
# ... per-stage commits ...
git push origin feat/nlinearscm-t       # open a PR ("Compare & pull request") for review
```
