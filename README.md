# CausalTemp-XAI

**Benchmarking Counterfactual Explanations for Temporal Causal Data**

[![CI](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml)

## Description

Standard CF metrics (validity, proximity, sparsity, OOD-plausibility) only
measure the relationship between a counterfactual and **the model** — none
measure its relationship to **the process that generated the data**.

`causaltemp-xai` closes that gap: it generates time series from a known SCM,
so a proposed counterfactual can be scored against what the true mechanism
would actually have produced.

**What it provides:**

- **A model-vs-world audit** comparing the method's intervention outcome
  against the true mechanism's outcome, decomposed into a trajectory term
  and an outcome term (exact, not Tian–Pearl bounded).
- **Do-complexity `D`** — how many timesteps a counterfactual must declare
  as interventions before the mechanism can reproduce it.
- **LinearSCM-T / NlinearSCM-T** — VAR(L) and additive-noise per-node MLP
  generators, each shipping true graph, true mechanism, and oracle
  abduction–action–prediction counterfactuals.
- **CFfaith** — a mechanism-consistency admission gate with two mutually
  exclusive semantics (`noiseless_rollout`, `pearl_delta`).
- **Seven wired CF methods** behind a uniform interface: `NoiselessSCMRecourse`
  (causal noiseless-rollout recourse), five reference methods from the
  vendored `cfts` repo (Wachter, COMTE, CONFETI, CounTS, CELS), and
  `TSCausalCF` (SCM-regularised, Bahri et al. 2025).

Findings are established on the synthetic tier, which is the only tier with
a ground-truth mechanism.

## Installation

All Python tooling goes through [`uv`](https://docs.astral.sh/uv/) (never pip):

```bash
git clone https://github.com/Healthpy/causaltemp-xai.git
cd causaltemp-xai
uv sync --extra dev        # creates .venv + uv.lock, installs deps + dev extras
```

## Reproduce the benchmark

```bash
# 0. environment
uv sync --extra dev

# 1. generate the locked paper dataset
uv run python -m causaltemp_xai.data_io --config full

# 2. train + freeze the LSTM classifier -> data/scm_t/full/lstm.pt
uv run python -m causaltemp_xai.classifiers.lstm --config full --train --patience 20

# 3. run the harness: CF methods x Axis-C + both CF-faith metrics
uv run python experiments/03_run_cf_methods.py --config full --n-cf 100 \
    --methods NoiselessSCMRecourse PearlSCMRecourse CftsWachter \
    CftsCOMTE CftsConfeti CftsCels TSCausal
uv run python experiments/04_evaluate_axes.py --config full

# 4. render the publication figures -> results/figures/
uv run python experiments/08_aggregate_and_report.py figures
```

Swap `--config full` for `--config smoke` (fast pass; what CI runs). Each
pipeline stage is a numbered script under `experiments/`, runnable
independently; outputs land under [`results/`](results/README.md). See
[`docs/05_evaluation_plan.md`](docs/05_evaluation_plan.md) for the
pre-registered hypotheses, and `experiments/*.py --help` for per-script
options (auxiliary methods, oracle control, seeds/CIs, horizon sweep, etc).

## Quickstart (library API)

```python
import numpy as np
from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.metrics import CFfaith, validity, proximity, sparsity

# 1. Generate synthetic causal time-series
gen = LinearSCMT(k=5, L=1, sparsity=0.2, noise_type="laplace", T=30, N=500, seed=0)
data = gen.generate()
X, Y = data["X"], data["Y"]

# 2. Train an LSTM classifier (sklearn-style wrapper; (N, T, k) in/out)
clf = LSTMClassifier(n_inputs=5, max_epochs=30)
clf.fit(X[:400], Y[:400], X[400:], Y[400:])

# 3. Evaluate causal faithfulness of a counterfactual
scorer = CFfaith(tol=1e-3, scale=1.0)
x_orig = X[0]
x_cf = X[0].copy()             # build your CF here

result = scorer.score(
    x_orig, x_cf,
    intervention_t=10,
    graph=data["graph"],
    mechanism=data["mechanism"],
)
print(result)   # {"hard": ..., "soft": ...}

# 4. Axis-C metrics
print("validity :", validity(x_cf[np.newaxis], clf, target_class=1))
print("proximity:", proximity(x_orig, x_cf, norm="l1"))
print("sparsity :", sparsity(x_orig, x_cf))
```

## Evaluation Axes

| Axis | Function / Class | Description |
|------|-----------------|-------------|
| **Validity** | `validity(x_cf, model, target_class)` | Flip-rate: fraction of CFs the model assigns to `target_class`. |
| **Proximity** | `proximity(x_original, x_cf, norm="l1")` | Mean L1 (or L2) distance between CF and original. Lower is better. |
| **Sparsity** | `sparsity(x_original, x_cf)` | Fraction of unchanged features in [0, 1]. Higher is sparser. |
| **OOD Plausibility** | `ood_plausibility(x_train, x_cf)` | IsolationForest decision score; higher means more in-distribution. |
| **CF-faith (hard/soft)** | `CFfaith.score(...)` | Whether/how closely the CF follows SCM mechanisms from the intervention time. |
| **Do-complexity** | `do_complexity(x, x_cf, mechanism)` | Timesteps the CF must declare as `do()` before the mechanism reproduces it. |
| **Model-vs-world gap** | `pns_direction(...)` | `Δ_total = Δ_trajectory + Δ_outcome`; always reported with `D`. |

`CFfaith` reports two mutually exclusive semantics side by side:
`"noiseless_rollout"` (default) and `"pearl_delta"`.

## Running Tests

```bash
uv run pytest tests/ -q
```

## Project Structure

```
causaltemp-xai/
├── causaltemp_xai/
│   ├── config.py                # BenchmarkConfig + SMOKE/FULL presets
│   ├── data_io.py               # split, generate/load datasets (+CLI)
│   ├── eval.py                  # evaluate_method (Axis-C + CF-faith) + shift_vr
│   ├── benchmarks/               # LinearSCM-T / NlinearSCM-T generators, mechanisms, oracle CF
│   ├── scm/                      # intervention helpers
│   ├── metrics/                  # taxonomy, axis_a/b/c, cf_faith, pns
│   ├── classifiers/lstm.py      # LSTM + sklearn-style wrapper
│   └── methods/counterfactual/  # scm_recourse, TSCausalCF, cfts-backed methods
├── third_party/                 # vendored cfts, dynamask, causalnex
├── experiments/                 # numbered phase scripts (01-11)
├── results/                     # figures + tables written by the pipeline
├── docs/general_plan.md         # the scientific claim, contributions, protocol
├── docs/05_evaluation_plan.md   # pre-registration: configs, methods, hypotheses
├── tests/
└── data/scm_t/                  # generated datasets + checkpoints (gitignored)
```

## License

MIT
