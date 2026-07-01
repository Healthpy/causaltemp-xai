# causal_tscf_bench

A research benchmark for evaluating the **causal faithfulness** of temporal counterfactual explanation (XAI) methods, grounded in Temporal Structural Causal Models (TSCMs).

---

## Overview

Existing time-series CF benchmarks measure validity, proximity, and sparsity — but none test whether a CF respects the ground-truth causal mechanism that generated the data. This benchmark fills that gap by:

1. **Generating synthetic time series** from analytically invertible TSCMs, making the true counterfactual X′_CF computable via the three-step causal ladder (abduction → action → prediction).
2. **Evaluating seven CF methods** from the [cfts_repo](../third_party/cfts_repo) library across four metric axes.
3. **Testing seven hypotheses** about when and why causal faithfulness degrades.

---

## Project Structure

```
causal_tscf_bench/
├── causal_tscf_bench/          # Main package
│   ├── scm/                    # Temporal SCM core
│   │   ├── dag.py              # LaggedDAG dataclass + sampler
│   │   ├── operators.py        # Invertible operator dictionary Φ
│   │   ├── tscm.py             # Forward simulation (Eq. 1)
│   │   ├── abduction.py        # Exact noise recovery (Eq. 2)
│   │   ├── intervention.py     # do-operator
│   │   └── counterfactual.py   # Causal ladder → X′_CF
│   ├── benchmarks/             # Benchmark datasets
│   │   ├── base.py             # BenchmarkDataset ABC + label threshold
│   │   ├── linear_scm_t.py     # LinearSCM-T (identity operators)
│   │   ├── nlinear_scm_t.py    # NlinearSCM-T (full invertible Φ)
│   │   ├── nlinear_ablations.py# Nonmonotonic + Regime-switching variants
│   │   └── sepsis_sim.py       # SepsisSim (MIMIC-IV / synthetic fallback)
│   ├── classifiers/            # Time-series classifiers
│   │   ├── base.py             # TSClassifier ABC
│   │   ├── lstm.py             # Bidirectional LSTM + attention  ← primary
│   │   ├── transformer.py      # CLS-token Transformer
│   │   └── tcn.py              # TCN (reference only; use LSTM for experiments)
│   ├── methods/                # XAI method wrappers
│   │   ├── base.py             # CFExplainer + AttributionMethod ABCs
│   │   ├── counterfactual/     # cfts-based CF wrappers (see Methods table)
│   │   │   ├── wachter.py      # WachterCF  — gradient baseline
│   │   │   ├── comte.py        # ComteCF    — instance substitution
│   │   │   ├── tsevo.py        # TSEvoCF    — evolutionary search
│   │   │   ├── glacier.py      # GlacierCF  — deep latent
│   │   │   ├── cels.py         # CELSCF     — saliency-guided
│   │   │   └── confetti.py     # ConfettiCF — NUN + genetic mask
│   │   ├── attribution/        # TimeSHAP, Dynamask
│   │   ├── concept/            # CBM-T, iVAE
│   │   └── causal/             # CARLA-Causal, CITRIS, iCITRIS
│   └── metrics/                # Four-axis evaluation
│       ├── axis_a.py           # ICC (attribution + latent), MCC (coverage + Hyvärinen), LD
│       ├── axis_b.py           # SHD, Lag Accuracy, AUC, TV-Confounding
│       ├── axis_c.py           # Validity, Proximity, Sparsity, OOD, TRSI, CF-faith, IVR
│       └── axis_d.py           # Shift-VR, Input Sensitivity, Concept Stability
├── experiments/                # Numbered experiment scripts
│   ├── 01_generate_benchmarks.py
│   ├── 02_train_classifiers.py
│   ├── 03_run_methods.py       # Runs all cfts CF methods → results/X_cf_*.npy
│   ├── 04_evaluate_axes.py     # Axis B + C evaluation → eval_*.json + CSV tables
│   ├── 05_ablations.py
│   ├── 06_hypothesis_tests.py  # H1–H7 with Wilcoxon signed-rank + BH correction
│   ├── 07_rank_correlation.py  # Spearman ρ: traditional metrics vs CF-faith
│   └── figures.py              # Fig 1–3 (scatter, boxplot, Spearman heatmap)
├── docs/
│   └── results_summary.md      # One-page narrative of H1–H7 findings
├── tests/
└── pyproject.toml
```

---

## Installation

```bash
cd causal_tscf_bench
pip install -e ".[dev]"
```

Dependencies: `torch>=2.3`, `numpy>=1.26`, `scipy>=1.12`, `scikit-learn>=1.4`, `tslearn>=0.6`, `pandas>=2.2`.

The CF methods delegate to `third_party/cfts_repo` (bundled as a git submodule). No extra install is required; the wrappers inject it into `sys.path` at runtime.

---

## Core Formalism

**Temporal SCM (Eq. 1)**

$$X_t^{(j)} := \sum_{i,\tau} w_{ij} \cdot \varphi_{ij}\!\left(X_{t-\tau}^{(i)}\right) + U_t^{(j)}$$

where $\varphi_{ij} \in \Phi = \{\text{id}, \sin, \cos, \tanh, |\cdot|, (\cdot)^2, e^{-|\cdot|}\}$.

**Abduction (Eq. 2)**

$$U_t^{(j)} = X_t^{(j)} - \sum_{i,\tau} w_{ij} \cdot \varphi_{ij}\!\left(X_{t-\tau}^{(i)}\right)$$

**CF-faith (Eq. 3)**

$$\text{CF-faith} = 1 - \frac{\text{DTW}(X'_{\text{exp}},\, X'_{\text{CF}})}{\text{DTW}(X,\, X'_{\text{CF}})}$$

where $X'_{\text{CF}}$ is the analytical ground-truth counterfactual. Higher is better; 0 = no gain over the factual; negative = worse than doing nothing.

---

## Quickstart

### 1. Generate a benchmark dataset

```bash
python experiments/01_generate_benchmarks.py --all
```

### 2. Train classifiers (LSTM and Transformer reach ≥ 90% val accuracy)

```bash
python experiments/02_train_classifiers.py --benchmark LinearSCMT --classifiers lstm transformer
```

### 3. Run all CF methods

```bash
python experiments/03_run_methods.py \
    --benchmark LinearSCMT \
    --classifier lstm \
    --methods wachter comte tsevo glacier cels confetti
```

### 4. Evaluate metric axes

```bash
# Single method
python experiments/04_evaluate_axes.py --benchmark LinearSCMT --classifier lstm --method wachter

# All methods (writes results/tables/table1_axis_c.csv and table2_axis_b.csv)
python experiments/04_evaluate_axes.py --benchmark LinearSCMT --classifier lstm --all_methods
```

### 5. Run hypothesis tests

```bash
python experiments/06_hypothesis_tests.py --results_dir results/
```

### 6. Rank-correlation analysis (H4 extended)

```bash
python experiments/07_rank_correlation.py --results_dir results/
```

### 7. Generate figures

```bash
python experiments/figures.py --results_dir results/ --out_dir results/figures/
```

---

## Benchmarks

| Benchmark | Operators | Noise | Identifiable | Notes |
|---|---|---|---|---|
| `LinearSCMT` | identity only | Laplace | Yes | Baseline linear SCM |
| `NlinearSCMT` | full Φ | Laplace | Yes | Nonlinear, exact abduction |
| `NlinearSCMT_Nonmonotonic` | Φ + non-invertible step | Laplace | No | Ablation for H4 |
| `NlinearSCMT_Regime` | full Φ, HMM weights | Laplace | Yes | Regime-switching, ablation for H7 |
| `SepsisSim` | clinical DAG | — | — | MIMIC-IV required; synthetic fallback available |

---

## Classifiers

| Classifier | Val Acc (LinearSCMT) | Val Acc (NlinearSCMT) | Meets ≥ 90% |
|---|---|---|---|
| LSTM | ~99% | ~97% | ✓ |
| Transformer | ~98% | ~97% | ✓ |
| TCN | ~89% | — | ✗ (use LSTM) |

LSTM is the recommended classifier for all experiments.

---

## Evaluation Axes

| Axis | Metrics | What it measures |
|---|---|---|
| **A — Concept Quality** | ICC (attribution mass + latent flip-rate), MCC (coverage + Hyvärinen optimal-assignment), LD | Attribution and disentanglement quality |
| **B — Graph Quality** | SHD, Lag Acc, AUC, TV-Conf | Causal graph recovery accuracy |
| **C — CF Quality** | Validity, Proximity, Sparsity, OOD, TRSI, CF-faith, IVR | Counterfactual correctness and causal faithfulness |
| **D — Robustness** | Shift-VR, InputSens, ConceptStab | Stability under distribution shift and perturbation |

**Key metrics (Axis C):**

| Metric | Formula | Notes |
|---|---|---|
| **CF-faith** | $1 - \text{DTW}(X'_{\text{exp}}, X'_{\text{CF}}) / \text{DTW}(X, X'_{\text{CF}})$ | Primary metric; higher = more causally faithful |
| **TRSI** | $\frac{1}{T-1}\sum_t \|\Delta_{t+1} - \Delta_t\|_2$ | L2 norm of delta-change per step; penalizes temporal discontinuity |
| **OOD** | Mean anomaly score from IsolationForest + LOF ensemble | Higher = more out-of-distribution |
| **IVR** | Fraction of CFs modifying time steps before the intervention point | 0 = respects temporal irreversibility |

**Key metrics (Axis A):**

| Metric | Formula | Notes |
|---|---|---|
| **ICC (latent)** | $\frac{1}{N}\sum_n \mathbf{1}[f(D_\psi(z^n + \delta e_i)) \neq f(x^n)]$ | Decoder intervention flip-rate |
| **MCC (Hyvärinen)** | Hungarian-assignment optimal mean absolute correlation | Matches true and predicted latent factors |

---

## XAI Methods Covered

All CF methods wrap [cfts_repo](../third_party/cfts_repo) implementations via a uniform `CFExplainer` interface (`fit(X_train, classifier)` / `explain(x, target_class, classifier)`).

| Class | Key | Paradigm | cfts function | Reference |
|---|---|---|---|---|
| `WachterCF` | `wachter` | Gradient baseline | `wachter_genetic_cf` | Wachter et al. (2017) |
| `ComteCF` | `comte` | Instance substitution | `comte_cf` | Delaney et al. (2021) |
| `TSEvoCF` | `tsevo` | Evolutionary search | `tsevo_cf` | Hollig et al. (2022) |
| `GlacierCF` | `glacier` | Deep latent | `glacier_cf` | Zhao & Doshi-Velez (2021) |
| `CELSCF` | `cels` | Saliency-guided | `m_cels_generate` | Li et al. (2024) |
| `ConfettiCF` | `confetti` | NUN + genetic mask | `confetti_genetic_cf` | Bahri et al. (2025) |
| `CARLACausal` | `carla` | Amortized causal | — | Karimi et al. (2021) |
| `TimeSHAP` | — | Attribution | — | Bento et al. (2021) |
| `Dynamask` | — | Attribution | — | Crabbe & van der Schaar (2021) |
| `CITRIS` | — | Causal disentanglement | — | Lippe et al. (2022) |
| `iCITRIS` | — | Causal disentanglement | — | Lippe et al. (2023) |

---

## Hypotheses Tested

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Causal CF methods achieve higher CF-faith than non-causal baselines | Wilcoxon signed-rank (paired) |
| H2 | Causal CF methods have lower IVR (respect temporal irreversibility) | Wilcoxon signed-rank (paired) |
| H3 | TRSI is lower for causal methods vs. gradient baseline (Wachter) | Wilcoxon signed-rank (paired) |
| H4 | CF-faith degrades under non-monotonic operators | Wilcoxon signed-rank |
| H5 | Laplace noise preserves class balance vs. Gaussian | Structural comparison |
| H6 | Label Visibility auto-tuning ensures balance in [0.3, 0.7] | Binomial test |
| H7 | CF-faith degrades under HMM regime-switching vs. stationary SCM | Structural comparison |

Multiple testing corrected via Benjamini-Hochberg FDR. Results are summarised in [`docs/results_summary.md`](docs/results_summary.md).

---

## Output Files

After a full run, the `results/` directory contains:

```
results/
├── LinearSCMT/lstm/
│   ├── X_cf_wachter.npy        # Counterfactuals (N, T, M)
│   ├── X_cf_comte.npy
│   ├── ...
│   ├── eval_wachter.json       # Axis B + C metrics
│   ├── eval_comte.json
│   └── all_methods_eval.json   # Combined table
├── tables/
│   ├── table1_axis_c.csv       # LaTeX-ready Axis C results
│   ├── table2_axis_b.csv       # LaTeX-ready Axis B results
│   └── table3_rank_corr.csv    # Spearman ρ table
├── rank_correlation.json
└── figures/
    ├── fig1_validity_vs_cffaith.pdf
    ├── fig2_cffaith_boxplot.pdf
    └── fig3_spearman_heatmap.pdf
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- SCM abduction round-trip (machine-precision noise recovery)
- Label balance guarantee across random seeds
- Metric edge cases (IVR = 0 for pre-intervention unchanged CFs, TRSI = 0 for constant shift)
- Classifier smoke tests (LSTM, Transformer)
- CF explainer and attribution method output shape and NaN checks

---

## Notes

- **cfts_repo**: All CF methods are thin wrappers around `third_party/cfts_repo`. The wrappers add the repo to `sys.path` at import time — no separate install needed.
- **CELSCF** expects channels-first `(M, T)` input internally; the wrapper handles the transpose automatically.
- **SepsisSim**: requires MIMIC-IV access via PhysioNet. Without credentials, `generate_synthetic()` provides a DAG-correct synthetic fallback.
- **CITRIS / iCITRIS**: `inferred_graph()` returns a `LaggedDAG` (shape `(M, M, max_lag)`); `infer_graph_instantaneous()` returns a dict of `LaggedDAG` objects (`lagged`, `instantaneous`, `combined`).

---

## Citation

If you use this benchmark, please cite the underlying method papers listed in the Methods table above.
