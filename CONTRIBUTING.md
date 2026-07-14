# Contributing to CausalTemp-XAI

Thank you for your interest in contributing! Here's how to set up your development environment.

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/Healthpy/causaltemp-xai.git
cd causaltemp-xai
uv sync --extra dev  # installs dependencies + dev tools
```

### 2. Install Pre-commit Hooks

Pre-commit hooks automatically enforce code quality on every commit:

```bash
pre-commit install
```

To run manually on all files:
```bash
pre-commit run --all-files
```

### 3. Code Quality Standards

The project uses:
- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **Ruff**: Fast linting (Python 3.9+)
- **MyPy**: Static type checking (optional in CI)

All configurations are in `pyproject.toml`.

#### Format code before commit:
```bash
black causaltemp_xai tests
isort causaltemp_xai tests
ruff check --fix causaltemp_xai tests
```

#### Run type checks:
```bash
mypy causaltemp_xai
```

### 4. Running Tests

```bash
uv run pytest tests/ -v           # all tests
uv run pytest tests/test_axis_c.py -v  # single file
uv run pytest --cov=causaltemp_xai --cov-report=html  # with coverage report
```

### 5. Reproducing Experiments

```bash
# Quick smoke test (k=5, T=30, N=500)
uv run python experiments/01_generate_benchmarks.py --config smoke
uv run python experiments/02_train_classifiers.py --config smoke
uv run python experiments/03_run_cf_methods.py --config smoke --n-cf 20
uv run python experiments/04_evaluate_axes.py --config smoke

# Full benchmark (k=10, T=100, N=10_000)
uv run python experiments/01_generate_benchmarks.py --config full
uv run python experiments/02_train_classifiers.py --config full
uv run python experiments/03_run_cf_methods.py --config full --n-cf 100
uv run python experiments/04_evaluate_axes.py --config full
```

See `docs/PIPELINE.md` for detailed pipeline documentation.

## Code Style Guidelines

- **Line length**: 100 characters (enforced by Black)
- **Docstrings**: NumPy format (see `causaltemp_xai/metrics/axis_c.py` for examples)
- **Type hints**: Encouraged for public APIs (gradual adoption)
- **Imports**: Sorted by isort, organized as: stdlib → third-party → local

## Commit Message Convention

Use clear, descriptive commit messages:
```
[component] brief description

- Detail 1
- Detail 2
```

Examples:
- `[metrics] add temporal sparsity decomposition`
- `[experiments] refactor phase 3 pipeline`
- `[docs] split README into separate doc files`

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and ensure tests pass: `uv run pytest tests/`
4. Ensure code quality: `pre-commit run --all-files`
5. Push to your fork
6. Open a pull request with clear description of changes

## Troubleshooting

**Pre-commit fails locally but CI passes?**
- Update hooks: `pre-commit autoupdate`
- Clear cache: `rm -rf .pre-commit-framework/`

**MyPy complaining about missing stubs?**
- Some third-party libraries don't have type stubs (torch, sklearn, etc.)
- Add `--ignore-missing-imports` or use `# type: ignore` comments sparingly

**Tests fail but CI passes?**
- Ensure you're using the same Python version as CI (check `.python-version` or `pyproject.toml`)
- Try: `uv venv` && `uv sync --extra dev`

## Questions?

See the project documentation in `docs/` or open an issue on GitHub.
