# CausalTemp-XAI

**Benchmarking Counterfactual Explanations for Temporal Causal Data**

[![CI](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml)

## Description


`causaltemp-xai` provides:

- **LinearSCM-T** - a VAR(L) benchmark generator with an explicit causal graph
  and non-Gaussian noise, so the true counterfactual distribution is known.
- **NlinearSCM-T** - the nonlinear sibling: same graph, additive-noise per-node
  MLP transition mechanisms (spectral-norm-capped for stability). Tests
  generalization beyond linear VAR identifiability. See below.
- **CFfaith** - a hard/soft metric that checks whether a proposed CF respects
  the causal mechanisms of the data-generating process.
- **Axis-C metrics** - four complementary quality axes (validity, proximity,
  sparsity, OOD plausibility) that together characterise a CF explanation.
- **Six wired CF methods** - `CARLARecourse` (causal noiseless-rollout
  recourse) plus five reference methods backed by the vendored `cfts` repo
  (Wachter, COMTE, CONFETI, CounTS, CELS), all behind a uniform interface.
  Native from-scratch `WachterCF`/`DiCECF` are also available via the library
  API.
- **Attribution foils** - integrated gradients + deletion/insertion
  perturbation curves.

## Installation

All Python tooling goes through [`uv`](https://docs.astral.sh/uv/) (never pip):

```bash
git clone https://github.com/Healthpy/causaltemp-xai.git
cd causaltemp-xai
uv sync --extra dev        # creates .venv + uv.lock, installs deps + dev extras
```

## Reproduce the benchmark

The full benchmark runs end-to-end from a clean checkout. Datasets and the
classifier checkpoint are **regenerated deterministically** (seeded), never
committed.

> **⚠ The v0.1 numbers are retracted, not merely stale.** They were produced
> with a TCN classifier (descoped 2026-07-08) and a since-deleted method
> (DiCE), on metric code predating the `99a2e9f` construct-validity fixes; the
> CF-faith column was then retracted outright by `d3c377d`. The historical
> record is kept at
> [`docs/archive/hypotheses_assessment.md`](docs/archive/hypotheses_assessment.md)
> **for provenance only — do not cite it.** Verdicts are regenerated into
> [`docs/05_evaluation_plan.md`](docs/05_evaluation_plan.md) §5 after the
> `full`/`full_nl` rerun.

```bash
# 0. environment
uv sync --extra dev

# 1. generate the locked paper dataset (k=10, L=1, T=100, N=10_000, Laplace noise)
uv run python -m causaltemp_xai.data_io --config full

# 2. train + freeze the LSTM classifier -> data/scm_t/full/lstm.pt
uv run python -m causaltemp_xai.classifiers.lstm --config full --train --patience 20

# 3. run the harness: CF methods x Axis-C + both CF-faith metrics
#    + IG attribution foil + Shift-VR-lite -> results/full/<classifier>/...
#    CftsCounts excluded from the current default run (PI decision, 2026-07-29)
uv run python experiments/03_run_cf_methods.py --config full --n-cf 100 \
    --methods CARLA PearlCARLA CftsWachter CftsCOMTE CftsConfeti CftsCels
uv run python experiments/04_evaluate_axes.py --config full

# 4. render the 3 publication figures -> results/figures/
uv run python experiments/06_aggregate_and_report.py figures
```

Swap `--config full` for `--config smoke` (k=5, T=30, N=500) for a fast pass; the
smoke pipeline is what CI runs. The scientific claim and evaluation protocol are
in [`docs/general_plan.md`](docs/general_plan.md); the pre-registered hypotheses
are in [`docs/05_evaluation_plan.md`](docs/05_evaluation_plan.md).

### Phased pipeline

Each stage of the pipeline above is its own numbered `uv run` script under
`experiments/`, so you can regenerate just the stage you're iterating on. All
outputs land under [`results/`](results/README.md) (figures **and** tables
included) — nothing is written into `experiments/`, which holds only the
runnable scripts:

```bash
uv run python experiments/01_generate_benchmarks.py --config smoke   # or --all
uv run python experiments/02_train_classifiers.py --config smoke     # linear configs only
uv run python experiments/03_run_cf_methods.py --config smoke --n-cf 20
uv run python experiments/04_evaluate_axes.py --config smoke
uv run python experiments/05_run_oracle_control.py --config smoke     # oracle positive control; any config, classifier-free

# phase 6 — post-hoc publication artifacts (two reports, one per subcommand)
uv run python experiments/06_aggregate_and_report.py figures                            # -> results/figures/
uv run python experiments/06_aggregate_and_report.py seeds --config smoke --seeds 0 1 2 # -> results/tables/ (multi-seed + bootstrap CIs)

# phase 7 — auxiliary method families outside the main 01-05 pipeline
uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method dynotears  # self-graphing + H3 graph-error split (nonlinear configs); --method citris for the secondary
uv run python experiments/07_auxiliary_methods.py --config smoke --method ivae          # decoder-based Axis-A ICC (iVAE latent traversal) with recon gate + latent->factor alignment
```

See [`results/README.md`](results/README.md) for the on-disk layout each
phase writes (`results/<config>/<classifier>/...`, `results/tables/`,
`results/figures/`).

## NlinearSCM-T (nonlinear benchmark)

`NlinearSCM-T` is the nonlinear sibling of LinearSCM-T: it keeps the same lagged
causal graph but replaces the VAR coefficients with **additive-noise per-node
MLP transitions**

```
x_t^i = decay_i · x_{t-1}^i  +  gain · tanh( MLP_i(masked lagged parents) )  +  eps_t^i
```

with spectral-norm-capped weights so trajectories stay bounded over long
horizons. It tests generalization **beyond linear VAR identifiability**. Because
the noise is **additive**, Pearl abduction is an exact subtraction
(`eps = x − f(parents)`), so `CFfaith` (both the rollout and the
abduction-based `pearl_delta` semantics) and the **oracle structural-CF** carry
over unchanged to the nonlinear mechanisms.

```bash
# 1. generate the nonlinear dataset (smoke: k=5, T=30, N=500 / full_nl mirrors `full`)
uv run python -m causaltemp_xai.data_io --config smoke_nl

# 2. run the harness on the nonlinear config
uv run python experiments/05_run_oracle_control.py --config smoke_nl     # or full_nl
```

Phase 05 is the **oracle-CF positive control** and is orthogonal to the
mechanism: it needs no classifier checkpoint and runs no real CF methods,
scoring CF-faith on the Stage-4 oracle structural counterfactual instead. Two
mutually-exclusive oracle variants are emitted — the Pearl oracle scores
`pearl_hard=1` and the noiseless (skeleton) oracle scores `rollout_hard=1` by
construction — demonstrating the rollout-vs-pearl contrast. It runs on **any**
config (`--config smoke` as readily as `smoke_nl`); the split between Phase
03/04 and Phase 05 is *explainer vs oracle*, **not** *linear vs nonlinear*.

**Scope:** this ships nonlinear *transitions* only. Nonlinear **mixing**
`x = g(z)` (an invertible observation map over latents) and **non-additive**
noise are a separate identifiability axis (iVAE/CITRIS) and are documented as a
**future extension** — see [`docs/general_plan.md`](docs/general_plan.md) §10
(Scope Boundaries) for why relaxing additivity would cost the exact abduction
the benchmark's contribution depends on.

## Quickstart (library API)

```python
import numpy as np
from causaltemp_xai.benchmarks.generator import LinearSCMT
from causaltemp_xai.classifiers import LSTMClassifier
from causaltemp_xai.metrics import CFfaith, validity, proximity, sparsity

# 1. Generate synthetic causal time-series
gen = LinearSCMT(k=5, L=1, sparsity=0.2, noise_type="laplace", T=30, N=500, seed=0)
data = gen.generate()
# data["X"]          shape (500, 30, 5)  -- multivariate time-series
# data["Y"]          shape (500,)        -- binary labels
# data["graph"]      shape (5, 5, 1)     -- adjacency per lag
# data["mechanism"]  Mechanism object    -- SCM transition (LinearMechanism here)

X, Y = data["X"], data["Y"]

# 2. Train an LSTM classifier (sklearn-style wrapper; (N, T, k) in/out)
clf = LSTMClassifier(n_inputs=5, max_epochs=30)
clf.fit(X[:400], Y[:400], X[400:], Y[400:])

# 3. Evaluate causal faithfulness of a counterfactual
scorer = CFfaith(tol=1e-3, scale=1.0)              # default noiseless-rollout semantics
x_orig = X[0]                  # shape (T, k)
x_cf   = X[0].copy()           # build your CF here

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
| **CF-faith (hard)** | `CFfaith.score(...)["hard"]` | 1.0 iff the CF exactly follows SCM mechanisms from the intervention time. |
| **CF-faith (soft)** | `CFfaith.score(...)["soft"]` | exp(-L1 residual / scale); continuous relaxation of hard faithfulness. |

Two CF-faith *semantics* are reported side by side (`CFfaith(semantics=...)`):
`"noiseless_rollout"` (default — what CARLA-causal is built to satisfy) and
`"pearl_delta"` (Pearl delta-recursion). A single CF cannot be `hard=1` under
both; the contrast is itself a benchmark result.

## Running Tests

```bash
uv run pytest tests/ -q
```

## Project Structure

```
causaltemp-xai/
├── causaltemp_xai/
│   ├── config.py                # BenchmarkConfig + SMOKE/FULL presets, shifted_config
│   ├── data_io.py               # stratified 60/20/20 split, generate/load datasets (+CLI)
│   ├── eval.py                  # evaluate_method (Axis-C + both CF-faith) + shift_vr
│   ├── benchmarks/
│   │   ├── generator.py         # LinearSCM-T VAR(L) + NlinearSCM-T MLP generators
│   │   ├── mechanisms.py        # Mechanism / LinearMechanism / MLPMechanism (+ serialize)
│   │   └── structural_cf.py     # oracle abduction-action-prediction counterfactual
│   ├── scm/
│   │   ├── intervention.py      # derive_intervention_t (uniform rule)
│   │   ├── abduction.py         # noise abduction for Pearl counterfactuals
│   │   └── counterfactual.py    # abduction-action-prediction machinery
│   ├── metrics/
│   │   ├── cf_faith.py          # CFfaith (noiseless_rollout | pearl_delta semantics)
│   │   ├── axis_c.py            # validity, proximity, sparsity, OOD, TRSI
│   │   └── axis_a|b|d.py        # concept, graph, robustness axis metrics
│   ├── classifiers/lstm.py      # LSTM + LSTMClassifier wrapper (+ train CLI)
│   └── methods/
│       ├── counterfactual/
│       │   ├── wachter.py       # WachterCF (gradient CF)
│       │   ├── dice.py          # DiCECF (dice-ml gradient + DPP fallback)
│       │   ├── carla.py         # CARLARecourse (causal noiseless-rollout recourse)
│       │   └── cfts_methods.py  # cfts-backed Wachter/COMTE/CONFETI/CounTS/CELS
│       ├── attribution/
│       │   ├── integrated_gradients.py   # hand-rolled IG attribution foil (WP3)
│       │   ├── perturbation_curves.py    # deletion / insertion curves
│       │   ├── timeshap.py      # TimeSHAP: official feedzai timeshap wrapper (Bento et al., 2021)
│       │   └── dynamask.py      # Dynamask: official Dynamask submodule wrapper (Crabbe & van der Schaar, 2021)
│       ├── concept/             # CBM-T probe + iVAE (experimental, unwired)
│       └── causal/              # DYNOTEARS + CITRIS (genuine, vendored)
├── third_party/cfts_repo/       # vendored cfts reference implementations
├── third_party/dynamask_repo/   # vendored official Dynamask
├── third_party/citris_repo/     # vendored official CITRIS (github.com/phlippe/CITRIS)
├── third_party/causalnex_repo/  # vendored McKinsey CausalNex (DYNOTEARS solver)
├── experiments/
│   ├── 01_generate_benchmarks.py    # phase 1: generate + persist datasets
│   ├── 02_train_classifiers.py      # phase 2: train the LSTM classifier
│   ├── 03_run_cf_methods.py         # phase 3: run CF methods + IG + shift-VR
│   ├── 04_evaluate_axes.py          # phase 4: Axis-C + CF-faith on persisted CFs
│   ├── 05_run_oracle_control.py     # phase 5: oracle structural-CF positive control (any config)
│   ├── 06_aggregate_and_report.py   # phase 6: multi-seed pooling + bootstrap CIs (M2) | figures
│   ├── 07_auxiliary_methods.py      # phase 7: self-graphing (Axis B) | iVAE ICC (Axis A)
│   └── _common.py                   # shared paths/loading for the phased pipeline
├── results/                     # figures + tables written by the phased pipeline
├── docs/general_plan.md         # the scientific claim, contributions, protocol
├── docs/05_evaluation_plan.md   # pre-registration: configs, methods, hypotheses
├── docs/risk_register.md        # descriptive risk register (RISK-01..RISK-14)
├── docs/archive/                # superseded memos, retracted numbers — do not cite
├── tests/
├── notebooks/                   # 01_data_exploration, 02_benchmark_exploration
└── data/scm_t/                 # generated datasets + checkpoints (gitignored)
```

## License

MIT
