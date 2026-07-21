# Development Guide

## Project Architecture

### Core Modules

#### `causaltemp_xai/scm/`
**Structural Causal Models** for temporal data.

- `generator.py`: Generates synthetic SCMs
  - `LinearSCM_T`: Vector AutoRegressive model (VAR)
  - `NlinearSCM_T`: Nonlinear variant with regime shifts
- `mechanisms.py`: Individual causal mechanisms and noise models
- `shift.py`: Distribution shift patterns for data augmentation

#### `causaltemp_xai/classifiers/`
**Time-series classifiers** for generating counterfactual instances.

- `lstm.py`: LSTM-based classifier
- `tcn.py`: Temporal Convolutional Network classifier
- Auto-selects best classifier based on data characteristics

#### `causaltemp_xai/methods/`
**Counterfactual explanation methods** (TSCF = Temporal Stepwise Counterfactuals).

Each method is a callable class taking `(x, t)` and returning `x_cf`:
- `dice_ml_wrapper.py`: Wrapper around Dice-ML (local LORE search)
- `gradient_cf.py`: Gradient-based counterfactuals (auto-diff through time)
- `exemplar_cf.py`: Nearest-neighbor exemplar lookup
- `oracle_linear.py`: Linear model oracle for benchmarking
- `oracle_nonlinear.py`: Nonlinear oracle using the true SCM

#### `causaltemp_xai/metrics/`
**Evaluation metrics** across four axes (Axis-C framework):

- `axis_c.py`: Validity, Proximity, Sparsity, OOD plausibility
  - **Sparsity**: Can now decompose into channel (features unchanged) and timepoint (timesteps unchanged) sparsity
- `axis_p.py`: Proximity-to-manifold (density estimation)
- `axis_f.py`: Faithfulness (feature importance alignment)
- Other metric utilities

#### `causaltemp_xai/benchmark/`
**Data generation and management** for reproducible experiments.

- `config.py`: Configuration loaders (YAML → dataclass)
- `dataset.py`: Dataset container with train/test splits
- `experiment_registry.py`: Centralized experiment tracking

### Experiment Pipeline

The workflow follows 6 numbered scripts in `experiments/`:

```
01_generate_benchmarks.py
  └─ Creates synthetic time-series data from SCM configs
  
02_train_classifiers.py
  └─ Trains LSTM/TCN classifiers on generated data
  
03_run_cf_methods.py
  └─ Generates counterfactuals using multiple methods
  
04_evaluate_axes.py
  └─ Evaluates on 4-axis framework (validity, proximity, sparsity, etc.)
  
05_run_oracle_control.py
  └─ Oracle structural-CF positive control (any config; no classifier, no explainer)
  
06_aggregate_and_report.py
  └─ Aggregates results and generates publication figures
```

Each script:
- ✅ Loads config from `configs/*.yaml`
- ✅ Uses seeded randomness for reproducibility
- ✅ Saves intermediate results to `results/`
- ✅ Supports parallelization via multiprocessing

### Configuration System

Experiment configs live in `configs/`:
- `linear_scm_t.yaml`: Linear VAR baseline
- `nlinear_scm_t.yaml`: Nonlinear benchmark
- `nlinear_scm_t_regime.yaml`: Regime-switching variant
- `nlinear_scm_t_nonmonotonic.yaml`: Non-monotonic responses

Each config specifies:
```yaml
scm:
  type: "linear" | "nlinear"
  k: 5                    # number of variables
  T: 30                   # time steps
  
data:
  N: 500                  # number of instances
  
classifier:
  type: "lstm" | "tcn"
  
cf_methods: ["dice_ml", "gradient_cf", "exemplar_cf"]
```

## Key Design Patterns

### 1. Temporal Metrics Decomposition
The `sparsity()` metric demonstrates how temporal structure should be handled:

```python
# Scalar mode (default, backward-compatible)
sparsity_overall = sparsity(x_orig, x_cf)  # float

# Detailed mode for temporal data
metrics = sparsity(x_orig, x_cf, return_detailed=True)
# Returns: {"channels": float, "timepoints": float}
# channel sparsity = fraction of features unchanged across all timesteps
# timepoint sparsity = fraction of timesteps unchanged across all features
```

**Why this matters**: A counterfactual changing only one feature at one timestep should score differently than one changing all features at one timestep. The decomposition captures this.

### 2. Method Interface Standardization
All CF methods inherit from a common interface:

```python
class CounterfactualMethod:
    def explain(self, x: np.ndarray, t: int) -> np.ndarray:
        """Generate counterfactual for instance x targeting class t."""
        pass
```

This enables:
- Easy swapping of methods in experiments
- Consistent input/output contracts
- Benchmark comparisons

### 3. Seeded Reproducibility
Every random operation uses a deterministic seed:

```python
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# Enables exact reproduction of:
# - SCM generation (same dynamics, same noise)
# - Train/test splits (same instances)
# - Model initialization (same weights)
# - CF generation (same search trajectories)
```

This is why experiment results are **not stored** in git—they're regenerated via the phased `experiments/01..07_*.py` pipeline (see `results/README.md`).

## Testing Strategy

Tests are organized by component:

```
tests/
├── test_scm.py              # LinearSCM_T, NlinearSCM_T
├── test_classifier.py       # LSTM, TCN training
├── test_methods.py          # CF method correctness
├── test_axis_c.py           # Metric calculations ✅ (19 passing)
├── test_axis_p.py           # Density metrics
├── test_axis_f.py           # Faithfulness metrics
├── test_config.py           # Config loading
├── test_experiment_registry.py  # Experiment tracking
└── test_stats.py            # Statistical tests
```

**Running tests**:
```bash
uv run pytest tests/ -v                    # all tests
uv run pytest tests/test_axis_c.py -v      # single file
uv run pytest --cov=causaltemp_xai         # with coverage
```

## Common Development Tasks

### Adding a New Counterfactual Method

1. Create `causaltemp_xai/methods/my_method.py`:
   ```python
   class MyMethod:
       def explain(self, x: np.ndarray, t: int) -> np.ndarray:
           # Your logic here
           return x_cf
   ```

2. Register in `causaltemp_xai/methods/__init__.py`

3. Add tests in `tests/test_methods.py`

4. Add to config `cf_methods` list in experiment configs

### Adding a New Metric

1. Create method in `causaltemp_xai/metrics/axis_c.py` (or appropriate file)
2. Follow NumPy docstring format
3. Add tests covering:
   - Basic case
   - Edge cases (zeros, identical arrays)
   - Tolerance parameter behavior
   - Temporal structure handling if applicable

### Modifying Experiment Pipeline

- **Sequential logic change**: Edit relevant numbered script (e.g., `03_run_cf_methods.py`)
- **Add new config parameter**: Update `configs/*.yaml` + `causaltemp_xai/benchmark/config.py`
- **Change output format**: Update both script AND `04_evaluate_axes.py` aggregation logic

## Performance Considerations

### Data Sizes
- Small (smoke test): k=5, T=30, N=500 (fast locally)
- Medium: k=10, T=100, N=5,000 (1-2 hours on CPU)
- Large: k=10, T=100, N=10,000 (full benchmark, 4-8 hours on GPU)

### Bottlenecks
1. **Classifier training**: ~40% of pipeline time
   - Solution: Use GPU (`torch.cuda.is_available()`)
2. **CF method search**: ~40% of pipeline time
   - Solution: Parallelize via multiprocessing pool
3. **Evaluation**: ~20% of pipeline time
   - Solution: Pre-compute manifold densities

### Optimization Tips
- Use `torch.no_grad()` during inference
- Cache classifier predictions
- Vectorize metric computations over batches
- Consider approximations for large N (e.g., approximate OOD via sampling)

## Debugging Workflows

### Issue: Results differ between runs

**Diagnosis**: Check if seed is set consistently:
```python
config.seed = 42  # Set explicitly
np.random.seed(config.seed)
torch.manual_seed(config.seed)
```

### Issue: OOD metric always returning 0.5

**Diagnosis**: Check IsolationForest training data:
```python
# oïf is trained on x_orig, checks if x_cf is anomalous
# If too few training samples, all scored as normal
print(f"OOD training samples: {len(X_train_for_oif)}")
```

### Issue: Counterfactuals not changing much

**Diagnosis**: Check method-specific hyperparameters:
```python
# DICE-ML: check loss weights
# Gradient-based: check learning rate and n_iterations
# Exemplar: check distance metric
```

See individual method files for tunable parameters.

## Documentation Files

- `README.md`: Quick start and high-level overview
- `CONTRIBUTING.md`: Dev setup, code style, testing (← You are here)
- `PIPELINE.md`: Detailed explanation of experiment workflow
- `docs/`: Technical deep-dives and API reference

For implementation details, see docstrings in each module.
