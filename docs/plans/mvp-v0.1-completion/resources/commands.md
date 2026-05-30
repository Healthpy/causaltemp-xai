# Commands

All Python runs go through **uv** (team convention — never pip/pipx).

## Environment (Stage 1)

```bash
# Initialise uv project + lockfile from pyproject.toml
uv sync --extra dev            # creates .venv + uv.lock, installs deps + dev extras
uv add dice-ml captum          # Stage 4 / Stage 6 additions
uv run python -c "import torch, sklearn, scipy; print('env ok')"
```

## Tests

```bash
uv run pytest tests/ -q                    # full suite
uv run pytest tests/test_generator.py -q   # single file
uv run pytest tests/ -q -k cf_faith        # by keyword
```

## Generate datasets (Stage 2)

```bash
uv run python -m causaltemp_xai.data_io --config smoke   # fast
uv run python -m causaltemp_xai.data_io --config full    # paper run
# writes data/linearscm_t/<config>/{X,Y}_{train,val,test}.npy + graph.npy + mechanisms.npz
```

## Train classifier (Stage 3)

```bash
uv run python -m causaltemp_xai.classifiers.tcn --config smoke --train
# saves checkpoint to data/linearscm_t/<config>/tcn.pt + logs test accuracy
```

## End-to-end experiment (Stage 8)

```bash
uv run python experiments/run_all.py --config smoke      # CI smoke
uv run python experiments/run_all.py --config full       # paper results
uv run python experiments/figures.py --results experiments/results.json
```

## CI (Stage 1)

CI should: `uv sync --extra dev` → `uv run pytest tests/` → (optionally) `uv run python experiments/run_all.py --config smoke` as a smoke run.

## Branch / commit

```bash
git checkout -b mvp-v0.1        # before Stage 1
# one commit per stage, conventional-commit messages from each stage file
```
