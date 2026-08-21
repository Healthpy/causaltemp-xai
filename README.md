# CausalTemp-XAI

**Benchmarking Counterfactual Explanations for Temporal Causal Data**

[![CI](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml)

## Description

Temporal counterfactual explanations are evaluated with validity, proximity,
sparsity and OOD-plausibility — every one of which measures a relationship
between the counterfactual and **the model**. None measures the relationship
between the counterfactual and **the process that generated the data**.

`causaltemp-xai` is a benchmark and audit protocol that closes that gap: it
generates time series from a known SCM, so a proposed counterfactual can be
scored against what the true mechanism would actually have produced. The field
uses structural causal models to *penalise* counterfactuals; nobody uses them to
*derive* the counterfactual the proposal is scored against.

**What it provides:**

- **A model-vs-world audit.** The method's own intervention is realised by the
  true mechanism and the two outcomes are compared, decomposed additively into
  a trajectory term and an outcome term. Necessity/sufficiency are computed
  **exactly**, not Tian–Pearl bounded, because additive noise makes abduction an
  exact subtraction.
- **Do-complexity `D`** — how many timesteps a counterfactual must declare as
  interventions before the mechanism can reproduce it. Pooled over 3 seeds on
  `full` (`T = 100`): a single-`do()` recourse control sits at `D = 1.0`, while
  the strongest published gradient method sits at `D = 96.4` and one method
  declares **every one of the 100 timesteps** an intervention. That is a
  trajectory rewrite, not an explanation.
- **Intervention-to-outcome distance as a reporting condition.** Effects on the
  label attenuate geometrically in the distance from the intervention to the
  label site, so recourse feasibility is a property of the *configuration*, not
  of the method — and validity numbers reported without that distance are
  uninterpretable.
- **LinearSCM-T / NlinearSCM-T** — VAR(L) and additive-noise per-node MLP
  generators, each shipping true graph, true mechanism, and oracle
  abduction–action–prediction counterfactuals. The **label functional** is a
  configurable axis, not a constant.
- **CFfaith** — a mechanism-consistency **admission gate** with two mutually
  exclusive semantics (`noiseless_rollout`, `pearl_delta`). A single CF cannot
  be hard-faithful under both; the contrast is itself a result. It is not a
  ranking axis — see [`docs/general_plan.md`](docs/general_plan.md) §7.
- **Seven wired CF methods** — `NoiselessSCMRecourse` (causal noiseless-rollout
  recourse) plus five reference methods backed by the vendored `cfts` repo
  (Wachter, COMTE, CONFETI, CounTS, CELS) and `TSCausalCF` (SCM-regularised,
  Bahri et al. 2025; wired into the default full-scale set 2026-08-06), all
  behind a uniform interface.

Findings are established on the synthetic tier, which is the **only** tier with
a ground-truth mechanism and therefore the only one where a metric can be shown
faithful. Real-signal tiers test whether the findings *travel*; they cannot
validate the metrics, and the docs are explicit about not conflating the two.

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

> **⚠ Also retracted (2026-08-03): `Δ_trajectory ≈ 1.0`.** The earlier headline
> "no wired method makes a causal intervention" was substantially an artifact of
> reading each counterfactual as a single `do()` at `t0`. Auditing the whole
> multi-timestep intervention a proposal implies collapses `Δ_trajectory` to
> 0.00 on every method. The finding is restated on do-complexity, which is not
> an artifact of the reading — see [`DECISIONS.md`](DECISIONS.md) and
> [`docs/general_plan.md`](docs/general_plan.md) §4.1.

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
#    + Shift-VR-lite -> results/full/<classifier>/...
#    CftsCounts excluded from the current default run (PI decision, 2026-07-29)
#    TSCausalCF (was CausalFeasibilityCF) wired into the default set 2026-08-06
uv run python experiments/03_run_cf_methods.py --config full --n-cf 100 \
    --methods NoiselessSCMRecourse PearlSCMRecourse CftsWachter \
    CftsCOMTE CftsConfeti CftsCels TSCausal
uv run python experiments/04_evaluate_axes.py --config full

# 4. render the 3 publication figures -> results/figures/
uv run python experiments/08_aggregate_and_report.py figures
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

# phase 8 — post-hoc publication artifacts (subcommands: figures | seeds | pns | horizon)
uv run python experiments/08_aggregate_and_report.py figures                            # -> results/figures/
uv run python experiments/08_aggregate_and_report.py seeds --config smoke --seeds 0 1 2 # -> results/tables/ (multi-seed + bootstrap CIs)

# phase 7 — auxiliary method families outside the main 01-05 pipeline
uv run python experiments/07_auxiliary_methods.py --config smoke_nl --method dynotears  # self-graphing + H3 graph-error split (nonlinear configs); --method pcmciplus for the second method; --method cross_method_agreement compares both

# model-vs-world audit + do-complexity (M2b). --schedule audits the whole
# multi-timestep intervention a CF implies rather than the single do() at t0.
uv run python experiments/07_auxiliary_methods.py --config full --method pns --seed 0
uv run python experiments/07_auxiliary_methods.py --config full --method pns --seed 0 --schedule
uv run python experiments/08_aggregate_and_report.py pns --config full   # -> results/tables/table_pns_do_complexity_full.csv
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
| **Do-complexity** | `do_complexity(x, x_cf, mechanism)` | Number of timesteps the CF must declare as `do()` before the mechanism reproduces it. `D = 0` is the vacuous case; `D → T` is a trajectory rewrite. |
| **Model-vs-world gap** | `pns_direction(...)` | `A`/`B`/`C` and the additive split `Δ_total = Δ_trajectory + Δ_outcome`. Always reported with `D`. |

Two CF-faith *semantics* are reported side by side (`CFfaith(semantics=...)`):
`"noiseless_rollout"` (default — what `NoiselessSCMRecourse` is built to satisfy) and
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
│   │   ├── labels.py            # label functionals (which scalar the label thresholds)
│   │   ├── mechanisms.py        # Mechanism / LinearMechanism / MLPMechanism (+ serialize)
│   │   └── structural_cf.py     # oracle abduction-action-prediction CF (single do() or schedule)
│   ├── scm/
│   │   └── intervention.py      # derive_intervention_t, is_vacuous_intervention (only file here)
│   ├── metrics/
│   │   ├── taxonomy.py          # AXES / AXIS_METRICS -- sole source of truth for A/B/C routing
│   │   ├── axis_a.py            # structure: SHD, lag accuracy, graph AUC (per dataset)
│   │   ├── axis_b.py            # robustness: Shift-VR, input sensitivity (per method)
│   │   ├── axis_c.py            # counterfactual quality: validity, proximity, sparsity, OOD, TRSI
│   │   ├── cf_faith.py          # CFfaith (noiseless_rollout | pearl_delta semantics)
│   │   └── pns.py               # model-vs-world audit: A/B/C split + do-complexity
│   ├── classifiers/lstm.py      # LSTM + LSTMClassifier wrapper (+ train CLI)
│   └── methods/
│       ├── counterfactual/
│       │   ├── scm_recourse.py         # Noiseless/Pearl SCM recourse positive controls
│       │   ├── causal_feasibility.py  # TSCausalCF (was CausalFeasibilityCF; Bahri et al. 2025, FISTA, SCM-regularised)
│       │   └── cfts_methods.py        # cfts-backed Wachter/COMTE/CONFETI/CounTS/CELS/NativeGuide
│       └── causal/              # DYNOTEARS (vendored) + PCMCIplus (tigramite, PyPI)
├── third_party/cfts_repo/       # vendored cfts reference implementations
├── third_party/dynamask_repo/   # vendored official Dynamask
├── third_party/causalnex_repo/  # vendored McKinsey CausalNex (DYNOTEARS solver)
├── experiments/
│   ├── 01_generate_benchmarks.py    # phase 1: generate + persist datasets
│   ├── 02_train_classifiers.py      # phase 2: train the LSTM classifier
│   ├── 03_run_cf_methods.py         # phase 3: run CF methods + shift-VR
│   ├── 04_evaluate_axes.py          # phase 4: Axis-C + CF-faith on persisted CFs
│   ├── 05_run_oracle_control.py     # phase 5: oracle structural-CF positive control (any config)
│   ├── 06_horizon_sweep.py          # phase 6: sweep t0, so the horizon claim is plotted not asserted
│   ├── 07_auxiliary_methods.py      # phase 7: self-graphing (Axis A) | PNS model-vs-world audit
│   ├── 07b_discovered_graph_real.py # phase 7b: discovered-graph CF-faith + cross-method agreement, Tier 2 (M4g/M4i)
│   ├── 08_aggregate_and_report.py   # phase 8: multi-seed pooling + bootstrap CIs | figures
│   ├── 09_tier1_synthetic_suite.py  # phase 9: Tier-1 suite, all 4 synthetic families x smoke/full (M4i)
│   ├── 10_tier2_real_suite.py       # phase 10: Tier-2 suite, 3 UCR/UEA real datasets (M4i)
│   ├── 11_tier3_real_suite.py       # phase 11: Tier-3 generic scaffold, known/discover/domain graph source (M4i)
│   └── _common.py                   # shared paths/loading for the phased pipeline
├── results/                     # figures + tables written by the phased pipeline
├── docs/general_plan.md         # the scientific claim, contributions, protocol
├── docs/05_evaluation_plan.md   # pre-registration: configs, methods, hypotheses
├── docs/risk_register.md        # descriptive risk register (RISK-01..RISK-19)
├── docs/archive/                # superseded memos, retracted numbers — do not cite
├── tests/
├── notebooks/                   # 01_data_exploration, 02_benchmark_exploration
└── data/scm_t/                 # generated datasets + checkpoints (gitignored)
```

## License

MIT
