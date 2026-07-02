# Refactoring Log — causaltemp_xai v0.1 → v0.2

## Overview

Large-scale structural refactoring to align causaltemp_xai with the
causal_tscf_bench reference structure. All working implementations are
preserved; the package is reorganised and extended with missing metrics.

---

## New Package: `scm/`

Ported from `causal_tscf_bench/scm/`:

| File | Source | Contents |
|------|--------|----------|
| `scm/__init__.py` | NEW | Package init with all SCM exports |
| `scm/dag.py` | NEW (bench port) | `LaggedDAG` dataclass + `sample_dag()` |
| `scm/operators.py` | NEW (bench port) | `OPERATORS`, `Mechanism` dataclass, `sample_mechanism()` |
| `scm/tscm.py` | NEW (bench port) | `simulate_tscm()`, `sample_noise()` |
| `scm/abduction.py` | NEW (bench port) | `abduct()` — Pearl abduction step |
| `scm/intervention.py` | COMBINED | `derive_intervention_t()` (MOVED from `methods/intervention.py`) + `Intervention` dataclass + `apply_intervention()` (bench port) |
| `scm/counterfactual.py` | NEW (bench port) | `compute_gt_counterfactual()`, `_forward_simulate_with_intervention()` |

**Note on two Mechanism types:**
- `scm/operators.py::Mechanism` = lightweight bench dataclass (operator name + params)
- `benchmarks/mechanisms.py::LinearMechanism`/`MLPMechanism` = causaltemp_xai model classes

Both coexist and serve different purposes.

---

## Renamed Package: `benchmark/` → `benchmarks/`

All files copied with updated imports (`causaltemp_xai.benchmark.` → `causaltemp_xai.benchmarks.`).

| Old file | New file | Change |
|----------|----------|--------|
| `benchmark/__init__.py` | `benchmarks/__init__.py` | Updated imports |
| `benchmark/mechanisms.py` | `benchmarks/mechanisms.py` | Import path only |
| `benchmark/structural_cf.py` | `benchmarks/structural_cf.py` | Import path only |
| `benchmark/generator.py` | `benchmarks/generator.py` | Import path only |

**New file:**
- `benchmarks/base.py` — `BenchmarkSplit` dataclass + `BenchmarkDataset` ABC (ported from bench, adapted to causaltemp_xai 60/20/20 split ratio and causaltemp_xai Mechanism types)
- `benchmarks/linear_scm_t.py` — `LinearSCMT` + `NlinearSCMT` now inheriting from `BenchmarkDataset`; `generate()` returns `BenchmarkSplit`; `generate_dict()` keeps backward-compat dict API

**Old directory deleted:** `benchmark/`

---

## Classifiers

| Change | Detail |
|--------|--------|
| DELETED | `classifiers/tcn.py` — TCN removed entirely |
| NEW | `classifiers/base.py` — `TSClassifier` ABC ported from bench |
| MODIFIED | `classifiers/__init__.py` — removed TCN exports, added `TSClassifier` |
| MODIFIED | `classifiers/lstm.py` — `LSTMClassifier` now inherits `TSClassifier`; existing method signatures unchanged |

---

## Methods Restructuring

### Counterfactual Methods (moved to subpackage)

| Old location | New location | Changes |
|--------------|--------------|---------|
| `methods/wachter.py` | `methods/counterfactual/wachter.py` | Added `fit()` / `explain()` aliases |
| `methods/dice.py` | `methods/counterfactual/dice.py` | Added `fit()` / `explain()` aliases |
| `methods/carla.py` | `methods/counterfactual/carla.py` | Added `fit()` / `explain()` + `set_causal_info()` |
| (from main) | `methods/counterfactual/cfts_methods.py` | MOVED |

**New file:** `methods/counterfactual/__init__.py`

**Old files deleted:** `methods/wachter.py`, `methods/dice.py`, `methods/carla.py`, `methods/cfts_methods.py`

### Attribution Methods (moved to subpackage)

| Old location | New location |
|--------------|--------------|
| `attribution/integrated_gradients.py` | `methods/attribution/integrated_gradients.py` |
| `attribution/perturbation_curves.py` | `methods/attribution/perturbation_curves.py` |

**New file:** `methods/attribution/__init__.py`

**Old directory deleted:** `attribution/` (top-level)

### New files

- `methods/base.py` — `CFExplainer` ABC + `AttributionMethod` ABC (ported from bench)

### Intervention moved

`methods/intervention.py` → `scm/intervention.py` (identical implementation)

Backward-compat re-export in `methods/__init__.py`:
```python
from causaltemp_xai.scm.intervention import derive_intervention_t
```

---

## Metrics Expansion

### axis_a.py (EXPANDED)

| Change | Detail |
|--------|--------|
| RENAMED | `icc()` → `icc_latent()` (model-based latent perturb metric) |
| NEW | `icc(attribution, int_channel)` — attribution-mass variant (ported from bench) |
| NEW | `mcc_concept(attribution, causal_parents)` — causal coverage |
| NEW | `latent_disentanglement(Z, X_channels)` — linear R² alignment |
| NEW | `compute_axis_a(...)` — aggregate over N instances |
| KEPT | `mig()`, `dci()`, `mcc()` unchanged |

### axis_b.py (NEW)

Full port from bench. Contains: `shd()`, `lag_accuracy()`, `graph_auc()`, `tv_confounding()`, `graph_error_decomposition()`, `compute_axis_b()`.

### axis_c.py (EXPANDED)

| Change | Detail |
|--------|--------|
| KEPT | `validity()`, `proximity()` (L1/L2), `sparsity()`, `ood_plausibility()` |
| NEW | `proximity_dtw()` — DTW-based proximity (bench port; requires tslearn) |
| NEW | `trsi()` — Temporal Relevance Smoothness Index |
| NEW | `ivr()` — Irreversibility-Violation-Rate |
| NEW | `ood_mahalanobis()` — Mahalanobis OOD fallback |
| NEW | `compute_axis_c()` — aggregate all Axis C metrics |

### axis_d.py (NEW)

Full port from bench. Contains: `shift_vr()` (noise-injection variant), `input_sensitivity()`, `concept_stability()`, `compute_axis_d()`.

**Note on two shift_vr functions:**
- `eval.shift_vr()` — benchmark-level validity-retention between base/shift environments (method comparison protocol)
- `metrics.axis_d.shift_vr()` — noise-injection variant (per-instance robustness protocol)

Both kept; they implement different protocols.

### cf_faith.py (UPDATED)

Import path: `causaltemp_xai.benchmark.mechanisms` → `causaltemp_xai.benchmarks.mechanisms`

---

## Import Updates

| File | Change |
|------|--------|
| `data_io.py` | `benchmark.generator` → `benchmarks.generator`; `benchmark.mechanisms` → `benchmarks.mechanisms` |
| `eval.py` | `methods.intervention` → `scm.intervention` |
| `metrics/cf_faith.py` | `benchmark.mechanisms` → `benchmarks.mechanisms` |
| `classifiers/__init__.py` | Removed TCN; added TSClassifier |
| `methods/__init__.py` | Re-structured to import from subpackages |
| `__init__.py` | Updated exports; version bumped to 0.2.0 |

---

## Redundancies Identified

1. **Two `compute_gt_counterfactual` implementations:**
   - `benchmarks/structural_cf.py::structural_counterfactual()` (causaltemp_xai original)
   - `scm/counterfactual.py::compute_gt_counterfactual()` (bench port)
   Both are kept; `scm/counterfactual.py` is the preferred API for new code.

2. **Two `shift_vr` functions** (documented above under axis_d.py).

3. **`benchmarks/generator.py` vs `benchmarks/linear_scm_t.py`:**
   `generator.py` is kept for backward compatibility; `linear_scm_t.py` is the new ABC-inheriting version. The `data_io.py` generator factory still uses `generator.py`; new code should prefer `linear_scm_t.py`.

---

## Deleted Files

- `classifiers/tcn.py`
- `methods/wachter.py`
- `methods/dice.py`
- `methods/carla.py`
- `methods/intervention.py`
- `attribution/` (entire directory, moved to `methods/attribution/`)
- `benchmark/` (entire directory, renamed to `benchmarks/`)
