# CausalTemp-XAI

**Benchmarking Counterfactual Explanations for Temporal Causal Data**

[![CI](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml)

## Description


`causaltemp-xai` provides:

- **LinearSCM-T** - a VAR(L) benchmark generator with an explicit causal graph
  and non-Gaussian noise, so the true counterfactual distribution is known.
- **CFfaith** - a hard/soft metric that checks whether a proposed CF respects
  the causal mechanisms of the data-generating process.
- **Axis-C metrics** - four complementary quality axes (validity, proximity,
  sparsity, OOD plausibility) that together characterise a CF explanation.
- **CF method stubs** - a uniform `generate(x, model)` interface for Wachter,
  DiCE, and CARLA-style recourse, ready for implementation or wrapping.

## Installation

```bash
git clone https://github.com/Healthpy/causaltemp-xai.git
cd causaltemp-xai
pip install -e .
```

For development dependencies (pytest, coverage):

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.classifiers import TCN, train_tcn
from causaltemp_xai.metrics import CFfaith, validity, proximity, sparsity

# 1. Generate synthetic causal time-series
gen = LinearSCMT(k=5, L=2, sparsity=0.3, noise_type="laplace", T=50, N=500, seed=42)
data = gen.generate()
# data["X"]          shape (500, 50, 5)  -- multivariate time-series
# data["Y"]          shape (500,)        -- binary labels
# data["graph"]      shape (5, 5, 2)     -- adjacency per lag
# data["mechanisms"] list of 2 ndarrays  -- VAR coefficient matrices

X, Y = data["X"], data["Y"]

# 2. Train a TCN classifier
model = train_tcn((X[:400], Y[:400]), target_acc=0.90)

# 3. Evaluate causal faithfulness of a counterfactual
scorer = CFfaith(tol=1e-3, scale=1.0)
x_orig = X[0]                  # shape (T, k)
x_cf   = X[0].copy()           # build your CF here

result = scorer.score(
    x_orig, x_cf,
    intervention_t=25,
    graph=data["graph"],
    mechanisms=data["mechanisms"],
)
print(result)   # {"hard": 1.0, "soft": 0.97}

# 4. Axis-C metrics
print("proximity:", proximity(x_orig, x_cf, norm="l1"))
print("sparsity :", sparsity(x_orig, x_cf))
```

## Evaluation Axes

| Axis | Function / Class | Description |
|------|-----------------|-------------|
| **Validity** | `validity(x_cf, model)` | Model prediction on the CF (checks class flip). |
| **Proximity** | `proximity(x_original, x_cf, norm="l1")` | Mean L1 (or L2) distance between CF and original. Lower is better. |
| **Sparsity** | `sparsity(x_original, x_cf)` | Fraction of unchanged features in [0, 1]. Higher is sparser. |
| **OOD Plausibility** | `ood_plausibility(x_train, x_cf)` | IsolationForest decision score; higher means more in-distribution. |
| **CF-faith (hard)** | `CFfaith.score(...)["hard"]` | 1.0 iff the CF exactly follows SCM mechanisms from the intervention time. |
| **CF-faith (soft)** | `CFfaith.score(...)["soft"]` | exp(-L1 residual / scale); continuous relaxation of hard faithfulness. |

## Running Tests

```bash
pytest tests/
```

## Project Structure

```
causaltemp-xai/
├── causaltemp_xai/
│   ├── benchmark/generator.py   # LinearSCM-T VAR(L) data generator
│   ├── metrics/
│   │   ├── cf_faith.py          # CFfaith class (hard + soft scores)
│   │   └── axis_c.py            # validity, proximity, sparsity, ood_plausibility
│   ├── classifiers/tcn.py       # TCN + train_tcn()
│   └── methods/
│       ├── wachter.py           # WachterCF stub
│       ├── dice.py              # DiCECF stub
│       └── carla.py             # CARLARecourse stub
├── experiments/run_all.py
├── tests/
├── notebooks/01_data_exploration.ipynb
└── data/linearscm_t/            # generated datasets (gitignored)
```

## License

MIT
