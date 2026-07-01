# 12-Week Execution Plan: Causal Temporal XAI Benchmark

**Project:** A four-axis evaluation framework for temporal counterfactual explainers grounded in Temporal Structural Causal Models (TSCM).  
**Root package:** `causal_tscf_bench/`  
**Governing plan:** `docs/trimmed_updated_general_plan.md`  
**Key references:** Runge 2023 (causal inference for time series), Pearl 2009 (SCM / do-calculus), Hyvärinen & Morioka 2019 (identifiability), Schulam & Saria 2017 (abduction–action–prediction).

---

## Overview

| Phase | Weeks | Theme |
|---|---|---|
| I — SCM Core | 1–2 | Generative engine: DAG, operators, simulation, causal ladder |
| II — Benchmarks | 3–4 | LinearSCM-T and NlinearSCM-T with label generation and analytical CFs |
| III — Classifiers | 5 | TCN, LSTM, Transformer trained to >90% accuracy |
| IV — Ablations + SepsisSim | 6 | Non-monotonic, regime-switching, clinical benchmark |
| V — Methods | 7–8 | All eight explainer families wrapped to a common interface |
| VI — Metrics | 9–10 | Four-axis evaluation suite (Axes A–D) |
| VII — Evaluation + Hypotheses | 11–12 | Full pipeline, H1–H7 tests, paper tables and figures |

---

## Week 1 — SCM Generative Engine

**Goal:** Implement DAG sampling, the invertible operator dictionary Φ, and the TSCM forward simulator (Eq. 1 of the plan).

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/scm/dag.py` | `LaggedDAG` dataclass; `sample_dag(n_channels, max_lag, edge_density, rng)` — samples an Erdős–Rényi random DAG over channel-lag pairs with acyclicity enforced by topological sort over instantaneous edges |
| `causal_tscf_bench/scm/operators.py` | `OPERATORS` dict mapping string keys to callables; `INVERTIBLE_OPERATORS` (excludes step); `sample_mechanism(dag, rng, sigma_w)` returns per-edge `(operator_key, weight)` pairs drawn as $w_{ij} \sim \mathcal{N}(0, \sigma^2_w)$ |
| `causal_tscf_bench/scm/tscm.py` | `simulate_tscm(dag, mechanisms, noise_seq, T, burn_in)` — implements Eq. 1: $X_t^{(j)} := \sum_{pa} w_{ij} \cdot \phi_{ij}(X_{t-\tau}^{(i)}) + U_t^{(j)}$ |
| `causal_tscf_bench/tests/test_scm_dag.py` | Acyclicity assertion; edge density recovery within ±20% of target; reproducibility under fixed seed |
| `causal_tscf_bench/tests/test_scm_operators.py` | Output shape; all operators finite on typical inputs; INVERTIBLE_OPERATORS excludes "step" |
| `causal_tscf_bench/tests/test_scm_tscm.py` | Burn-in convergence; output shape `(N, T, M)`; zero-graph case degenerates to pure noise |

### Acceptance Criteria
- `pytest tests/test_scm_*.py` passes with zero failures.
- Simulated sequences are stationary for linear operators with spectral radius < 1 (verified by sample autocorrelation decay).

### Dependencies
- NumPy ≥ 1.26, SciPy ≥ 1.12 (for Laplace / uniform noise sampling).

### Literature
- Runge et al. (2023), *Causal inference for time series*, Nature Reviews Earth & Environment — defines TSCM and lagged DAG structure used in Eq. 1.

---

## Week 2 — Causal Ladder: Abduction, Action, Prediction

**Goal:** Implement the three-step causal ladder to produce the analytical ground-truth counterfactual $X'_{CF}$ (Eq. 2 of the plan).

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/scm/abduction.py` | `abduct(X, dag, mechanisms) -> U` — exact inversion of Eq. 1: $U_t^{(j)} = X_t^{(j)} - \sum_{pa} w_{ij} \cdot \phi_{ij}(X_{t-\tau}^{(i)})$ |
| `causal_tscf_bench/scm/intervention.py` | `do_intervention(dag, int_channel, int_time, int_value) -> ModifiedDAG` — replaces structural equation of target node with constant, severs incoming edges at `int_time` |
| `causal_tscf_bench/scm/counterfactual.py` | `compute_gt_counterfactual(X, dag, mechanisms, int_channel, int_time, int_value) -> X_cf` — chains abduction → action → prediction; also exposes `forward_simulate(dag, mechanisms, U, override)` for downstream use |
| `causal_tscf_bench/tests/test_scm_causal_ladder.py` | Round-trip test: generate X → abduct U → forward-simulate with no intervention → recover X within 1e-8 tolerance; intervention propagation test: perturb root node, verify causal descendants change, verify non-descendants unchanged |

### Acceptance Criteria
- Round-trip reconstruction error < 1e-8 (machine precision) for all operator types in INVERTIBLE_OPERATORS.
- Intervention on a root node with no causal parents changes all downstream variables and leaves causally disconnected variables unchanged.

### Literature
- Pearl (2009), *Causality* (2nd ed.), Ch. 7 — formal definition of abduction–action–prediction for structural CFs.
- Schulam & Saria (2017), *Reliable Decision Support using Counterfactual Models* — motivates exact abduction in additive SCMs.

---

## Week 3 — LinearSCM-T Benchmark

**Goal:** Generate and package the LinearSCM-T dataset (identity operators, VAR(1) mechanics) including labels via the Label Visibility principle and pre-computed ground-truth CFs.

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/benchmarks/base.py` | `BenchmarkDataset` ABC — defines `generate()`, `load()`, `get_gt_counterfactuals(idx, int_channel, int_time, int_value)` interface |
| `causal_tscf_bench/benchmarks/linear_scm_t.py` | `LinearSCMT(BenchmarkDataset)` — generates N=10,000 sequences; k ∈ {5, 10} channels; L=1 lag; sparsity s/k² ∈ {0.1, 0.2}; noise ∈ {Laplace, Uniform, Gaussian (ablation)}; T ∈ {50, 100}; labels via $Y = \mathbb{1}[X_T^{(j^*)} > \theta]$ with θ auto-tuned for balance ∈ [0.3, 0.7]; saves X, Y, graph.npy, mechanisms.npz, meta.json, cf_trajectories.npz |
| `causal_tscf_bench/configs/linear_scm_t.yaml` | Full configuration matrix for all LinearSCM-T sub-experiments |
| `causal_tscf_bench/experiments/01_generate_benchmarks.py` | CLI entry-point: `python 01_generate_benchmarks.py --config configs/linear_scm_t.yaml --out data/linear_scm_t/` |
| `causal_tscf_bench/tests/test_benchmark_linear.py` | Shape assertions; class balance in [0.3, 0.7]; GT CF shape matches X; Gaussian ablation config togglable |

### Acceptance Criteria
- All 8 LinearSCM-T configurations (2 sizes × 2 densities × 2 T values) generate without errors.
- Class balance [0.3, 0.7] achieved across all configs.
- GT CFs pass the round-trip test from Week 2 (residuals < 1e-6).

### Literature
- Hyv‌ärinen & Morioka (2019), *Nonlinear ICA of Temporally Dependent Stationary Sources* — justifies non-Gaussian noise as identifiability condition for the Gaussian ablation comparison in H5.
- Hauser & Bühlmann (2012), *Characterization and Greedy Learning of Interventional Markov Equivalence Classes* — motivates sparse DAG sampling at s/k² ∈ {0.1, 0.2}.

---

## Week 4 — NlinearSCM-T Benchmark (Base Config)

**Goal:** Extend the pipeline to nonlinear mechanisms drawn from Φ, verify exact abduction still holds, and produce the NlinearSCM-T base dataset.

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/benchmarks/nlinear_scm_t.py` | `NlinearSCMT(BenchmarkDataset)` — same DAG sampling as Week 3; mechanisms drawn from `INVERTIBLE_OPERATORS` (no step function); same label scheme; same artifact format |
| `causal_tscf_bench/configs/nlinear_scm_t.yaml` | Config for base NlinearSCM-T; same dimensional grid as LinearSCM-T |
| `causal_tscf_bench/tests/test_benchmark_nlinear.py` | Operator diversity check (at least 3 distinct operators per dataset); exact abduction round-trip < 1e-6; label balance |

### Acceptance Criteria
- Abduction round-trip residuals < 1e-6 for all seven invertible operators.
- Operator dictionary coverage: at least 5 of 7 operators used in a dataset with k=10 channels.

### Literature
- Zhang & Hyvärinen (2009), *On the Identifiability of the Post-Nonlinear Causal Model* — establishes why non-Gaussian noise enables identifiability with nonlinear mechanisms; informs why the Gaussian ablation fails on NlinearSCM-T.

---

## Week 5 — Classifier Training

**Goal:** Train TCN, LSTM, and Transformer classifiers on LinearSCM-T and NlinearSCM-T base configs to ≥90% test accuracy. Freeze these classifiers for all downstream explainer evaluation.

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/classifiers/base.py` | `TSClassifier` ABC — `fit(X_train, Y_train)`, `predict(X)`, `predict_proba(X)`, `save(path)`, `load(path)` |
| `causal_tscf_bench/classifiers/tcn.py` | `TCNClassifier` — PyTorch TemporalConvNet with residual dilated convolutions (Bai et al. 2018) |
| `causal_tscf_bench/classifiers/lstm.py` | `LSTMClassifier` — bidirectional LSTM with attention pooling |
| `causal_tscf_bench/classifiers/transformer.py` | `TransformerClassifier` — patch-based encoder with CLS token classification head |
| `causal_tscf_bench/experiments/02_train_classifiers.py` | CLI: trains all three classifiers on a specified benchmark config; saves `.pt` weights and test accuracy JSON |
| `causal_tscf_bench/tests/test_classifiers.py` | Forward pass shape; prediction sum to 1; save/load round-trip preserves output |

### Acceptance Criteria
- All three classifiers reach ≥90% test accuracy on LinearSCM-T `k=5, T=50, density=0.1` within 30 epochs.
- Saved weights reload identically (max output difference < 1e-6).

### Literature
- Bai et al. (2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* — TCN architecture reference.
- Vaswani et al. (2017), *Attention Is All You Need* — Transformer architecture reference.

---

## Week 6 — NlinearSCM-T Ablations + SepsisSim Stub

**Goal:** Add non-monotonic and regime-switching sub-configurations to NlinearSCM-T. Stand up the SepsisSim data loading skeleton.

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/benchmarks/nlinear_ablations.py` | `NlinearSCMT_Nonmonotonic` — adds step function to operator pool (making exact abduction undefined; uses approximate noise for CF baseline); `NlinearSCMT_Regime` — HMM with R ∈ {2, 3} regimes switching active weights at random change-points |
| `causal_tscf_bench/benchmarks/sepsis_sim.py` | `SepsisSim(BenchmarkDataset)` — loads clinician-validated DAG from `configs/sepsis_dag.json`; Neural-ODE state transitions (torchdiffeq); time-varying treatment variables; MAP threshold label; stub: `generate()` raises `NotImplementedError` with roadmap comment until MIMIC-IV access is confirmed |
| `causal_tscf_bench/configs/nlinear_scm_t_nonmonotonic.yaml` | Operator pool includes "step"; ablation metadata flags `identifiable: false` |
| `causal_tscf_bench/configs/nlinear_scm_t_regime.yaml` | HMM parameters: n_regimes ∈ {2, 3}, transition_prob ∈ {0.05, 0.1}, min_regime_len=10 |
| `causal_tscf_bench/tests/test_benchmark_ablations.py` | Regime-switching: segment lengths respect `min_regime_len`; weight matrices differ across regimes; non-monotonic: step operator appears in sampled mechanisms |

### Acceptance Criteria
- Regime-switching datasets have detectable structural breaks verifiable by CUSUM test (p < 0.01).
- SepsisSim skeleton imports and instantiates without error (generate raises `NotImplementedError`).

### Literature
- Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series* — HMM regime-switching model.
- Runge et al. (2019), *Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets* — motivates PCMCI as upstream discovery tool for regime-identified systems.

---

## Week 7 — CF Method Wrappers (All Four Paradigms)

**Goal:** Wrap CoMTE, TSEvo, Glacier, and DiCE (four paradigms + naive baseline) behind a common `CFExplainer` interface. Wire them to the `third_party/cfts_repo` submodule where implementations exist.

### Deliverables

| File | Paradigm | Source |
|---|---|---|
| `causal_tscf_bench/methods/base.py` | — | Abstract `CFExplainer` ABC: `explain(x, target_class, classifier) -> X_cf`; `fit(X_train, classifier)` |
| `causal_tscf_bench/methods/counterfactual/comte.py` | Instance substitution | Wraps `third_party/cfts_repo/cfts/cf_comte/comte.py` |
| `causal_tscf_bench/methods/counterfactual/tsevo.py` | Evolutionary | Wraps `third_party/cfts_repo/cfts/cf_tsevo/tsevo.py` |
| `causal_tscf_bench/methods/counterfactual/glacier.py` | Deep latent | Wraps `third_party/cfts_repo/cfts/cf_glacier/glacier.py` |
| `causal_tscf_bench/methods/counterfactual/dice.py` | Naive baseline | Wraps `third_party/cfts_repo/cfts/cf_wachter/wachter.py` with DiCE-style diversity |
| `causal_tscf_bench/experiments/03_run_methods.py` | — | CLI: runs all four CF methods on a specified benchmark split; writes `results/{method}_{benchmark}_{classifier}.npz` |
| `causal_tscf_bench/tests/test_methods_cf.py` | — | Output shape matches input X; valid class predicted by classifier; runtime < 60s per 50 instances |

### Acceptance Criteria
- All four wrappers produce `X_cf` with shape `(N, T, M)` matching the input.
- `f(X_cf)` returns the target class for ≥50% of instances on LinearSCM-T (pre-check that methods can converge).

### Literature
- Delaney et al. (2021), *Instance-Based Counterfactual Explanations for Time Series Classification* — CoMTE.
- Hollig et al. (2022), *TSEvo: Evolutionary Counterfactual Explanations for Time Series* — TSEvo.
- Zhao & Doshi-Velez (2021), *Glacier: Guided Locally Constrained Counterfactual Explanations* — Glacier.
- Wachter et al. (2017), *Counterfactual Explanations Without Opening the Black Box* — DiCE baseline.

---

## Week 8 — Attribution, Concept, and Causal Method Wrappers

**Goal:** Wrap all remaining method families (attribution, concept-based, causal/graph-aware) behind the same interface. These methods produce attributions or concept scores, not CFs — adapters expose scores for Axes A and B.

### Deliverables

| File | Family | Interface |
|---|---|---|
| `causal_tscf_bench/methods/attribution/timeshap.py` | Attribution | `attribute(x, classifier) -> phi` shape `(T, M)` |
| `causal_tscf_bench/methods/attribution/dynamask.py` | Attribution | Same as TimeSHAP; wraps Crabbe & van der Schaar (2021) |
| `causal_tscf_bench/methods/concept/cbm_t.py` | Concept-based | `fit(X, Y)`, `predict_concepts(x) -> z` latent; `concept_to_class(z) -> y` |
| `causal_tscf_bench/methods/concept/ivae.py` | Concept-based | `fit(X, S)` where S is auxiliary variable (regime/label); `encode(x) -> z`; `decode(z) -> x_hat` |
| `causal_tscf_bench/methods/causal/carla_causal.py` | Amortized/controllable | `CFExplainer` subclass; mask + magnitude heads |
| `causal_tscf_bench/methods/causal/citris.py` | Causal/graph-aware | `CFExplainer` subclass; also exposes `inferred_graph() -> LaggedDAG` for Axis B |
| `causal_tscf_bench/methods/causal/icitris.py` | Causal/graph-aware | Same as CITRIS with instantaneous causal effects |
| `causal_tscf_bench/tests/test_methods_attribution.py` | — | Attribution shape; finite values; time-summed attribution > 0 for informative features |
| `causal_tscf_bench/tests/test_methods_concept.py` | — | Concept space dimensionality; reconstruction loss finite |
| `causal_tscf_bench/tests/test_methods_causal.py` | — | CF output shape; inferred_graph returns LaggedDAG; SHD computable |

### Acceptance Criteria
- All attribution methods produce finite `(T, M)` tensors.
- CITRIS / iCITRIS `inferred_graph()` returns a `LaggedDAG` compatible with Axis B scoring.

### Literature
- Koh et al. (2020), *Concept Bottleneck Models* — CBM-T.
- Khemakhem et al. (2020), *Variational Autoencoders and Nonlinear ICA: A Unifying Framework* — iVAE; also provides the identifiability conditions for Axis A ICC scoring.
- Löwe et al. (2022), *CITRIS: Causal Identifiability from Temporal Interventional Sequences* — CITRIS / iCITRIS.
- Crabbe & van der Schaar (2021), *Explaining Time Series Predictions with Dynamic Masks* — Dynamask.

---

## Week 9 — Metrics: Axis A (Concept Quality) and Axis B (Graph Quality)

**Goal:** Implement the full Axis A and Axis B metric suites.

### Deliverables

| File | Metric | Formula / Source |
|---|---|---|
| `causal_tscf_bench/metrics/axis_a.py` | `icc(method, decoder, X, f, delta, n_samples)` | $\mathrm{ICC}_i = \frac{1}{N}\sum_n \mathbb{1}[f(D_\psi(z^n + \delta_i e_i)) \neq f(x^n)]$ |
| | `mcc(z_true, z_pred)` | Mean Correlation Coefficient (Hyvärinen 2019) |
| | `latent_disentanglement(z, u_true)` | Mutual information gap between $z$ dimensions and true $U$ channels |
| `causal_tscf_bench/metrics/axis_b.py` | `shd(graph_true, graph_pred)` | Structural Hamming Distance on lagged adjacency |
| | `lag_accuracy(graph_true, graph_pred)` | Fraction of detected edges with correct τ |
| | `auc_roc_graph(graph_true, edge_probs)` | AUC on edge probability ranking |
| | `tv_confounding_robustness(method, X, Y, time_varying_confounders)` | False attribution rate under known time-varying hidden drivers |
| | `graph_error_decomposition(method, X, graph_true, mechanisms)` | Returns `(cf_faith_vs_gt, cf_faith_vs_inferred)` tuple for CITRIS/iCITRIS |
| `causal_tscf_bench/tests/test_metrics_ab.py` | — | SHD=0 for perfect graph; ICC=1 for perfect decoder intervention; MCC=1 for identity mapping |

### Acceptance Criteria
- SHD(G, G) = 0 and SHD(G, empty_graph) = number of edges in G for all test cases.
- ICC returns values in [0, 1]; MCC returns values in [-1, 1].

### Literature
- Peters et al. (2017), *Elements of Causal Inference* — SHD definition and interpretation.
- Hyvärinen & Morioka (2019) — MCC definition for nonlinear ICA evaluation.

---

## Week 10 — Metrics: Axis C (CF Quality) and Axis D (Robustness)

**Goal:** Implement the full Axis C and Axis D metric suites, including the normalized DTW CF-faith formula and TRSI.

### Deliverables

| File | Metric | Formula |
|---|---|---|
| `causal_tscf_bench/metrics/axis_c.py` | `validity(X_cf, classifier, target_class)` | $\mathbb{1}[f(X'_{exp}) = y^*]$ |
| | `proximity(X_cf, X, p)` | $\lVert X'_{exp} - X \rVert_p$ for p ∈ {1, 2} |
| | `sparsity(X_cf, X)` | $L_0$ fraction: $\frac{1}{T \cdot M} \lVert X'_{exp} - X \rVert_0$ |
| | `ood_plausibility(X_cf, X_train)` | IF and LOF scores from sklearn |
| | `trsi(X_cf, X)` | $\frac{1}{T-1}\sum_t \lVert \Delta_{t+1} - \Delta_t \rVert_2$ where $\Delta_t = X'_t - X_t$; lower = smoother |
| | `cf_faith_soft(X_cf, X_gt_cf, X)` | $1 - \frac{\mathrm{DTW}(X'_{exp}, X'_{CF})}{\mathrm{DTW}(X, X'_{CF})}$ |
| | `cf_faith_hard(X_cf, X_gt_cf, epsilon)` | $\mathbb{1}[\mathrm{DTW}(X'_{exp}, X'_{CF}) < \epsilon]$ |
| | `ivr(X_cf, X, int_time)` | Fraction of CFs modifying time steps $t < \mathcal{T}_{int}$ |
| `causal_tscf_bench/metrics/axis_d.py` | `shift_vr(method, X_test_shift, classifier, target_class)` | Validity on regime-switched held-out split |
| | `input_sensitivity(method, X, classifier, eps)` | Lipschitz estimate: $\sup \lVert \phi(x+\delta) - \phi(x) \rVert / \lVert \delta \rVert$ |
| | `concept_stability(method, X, Y, R)` | Jaccard of top-k concepts across R retraining restarts |
| `causal_tscf_bench/tests/test_metrics_cd.py` | — | CF-faith=1 when X_cf=X_gt_cf; CF-faith≤0 when X_cf=X; IVR=0 when int_time=0; TRSI=0 for constant delta |

### Acceptance Criteria
- `cf_faith_soft(X_gt_cf, X_gt_cf, X)` returns 1.0 for all test instances.
- `ivr(X_cf=X, int_time=T//2)` returns 0.0 (original sequence never violates irreversibility).
- `trsi` is monotonically larger for randomly perturbed sequences vs. smooth linear interpolations.

### Literature
- Sakoe & Chiba (1978), *Dynamic Programming Algorithm Optimization for Spoken Word Recognition* — DTW definition used in CF-faith.
- Guidotti et al. (2022), *Counterfactual Explanations and How to Find Them* — proximity, sparsity, validity definitions.
- Delaney et al. (2021) — temporal coherence motivation for TRSI.

---

## Week 11 — Evaluation Harness and Hypotheses H1–H5

**Goal:** Wire all components into a reproducible end-to-end evaluation pipeline. Test hypotheses H1–H5 and produce preliminary result tables.

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/experiments/04_evaluate_axes.py` | CLI: loads benchmark + classifier + method results; runs all four axes; writes `results/{method}_{benchmark}_{classifier}_axes.json` |
| `causal_tscf_bench/experiments/06_hypothesis_tests.py` | H1: check CF-faith < 0.3 for DiCE/CoMTE/TSEvo; H2: check ICC < 0.5 for CBM-T/iVAE vs. causal axis; H3: check CF-faith > 0.7 for CITRIS; H4: Spearman ρ between validity/proximity ranking and CF-faith ranking; H5: CF-faith on non-Gaussian vs. Gaussian ablation configs |
| `causal_tscf_bench/experiments/figures.py` | Fig 1: scatter(validity, CF-faith) per method per benchmark; Fig 2: CF-faith distribution boxplot; Fig 3: Spearman ρ heatmap (H4) |
| `results/tables/` | `table1_axis_c.csv`, `table2_axis_b.csv`, `table3_axis_a.csv` — LaTeX-ready |
| `causal_tscf_bench/tests/test_evaluation_harness.py` | Smoke test: full pipeline on smoke config (N=200, T=20, k=3) completes in < 120s |

### Acceptance Criteria
- Smoke test passes end-to-end without errors.
- H1–H5 outputs include p-values from Wilcoxon signed-rank tests for paired comparisons.
- Fig 1–3 render without error.

### Literature
- Fawcett (2006), *An Introduction to ROC Analysis* — AUC-ROC for graph quality (Axis B).
- Wilcoxon (1945) — non-parametric test for H1–H5 comparisons across method pairs.

---

## Week 12 — Ablations H6–H7, SepsisSim Full Run, Final Tables

**Goal:** Run NlinearSCM-T ablations for H6 (non-monotonic), regime-switching for H7, complete SepsisSim if MIMIC-IV access is available, and finalize all paper tables and figures.

### Deliverables

| File | Content |
|---|---|
| `causal_tscf_bench/experiments/05_ablations.py` | H6: compare Glacier CF-faith on base vs. non-monotonic NlinearSCM-T; H7: compare CoMTE vs. CITRIS Shift-VR on regime-switching NlinearSCM-T |
| `results/tables/table4_ablations.csv` | H6 and H7 results with bootstrap 95% CIs (B=1000) |
| `results/figures/fig4_h6_nonmonotonic.png` | CF-faith vs. operator type distribution for Glacier |
| `results/figures/fig5_h7_shift_vr.png` | Shift-VR bar chart: CoMTE vs. CITRIS across regime configs |
| `results/sepsis_sim/` | SepsisSim results if MIMIC-IV available; otherwise synthetic SepsisSim with fixed clinical DAG |
| `causal_tscf_bench/experiments/07_rank_correlation.py` | H4 final: Spearman ρ between all traditional-metric rankings and CF-faith / ICC rankings; report p-values |
| `docs/results_summary.md` | One-page narrative of all seven hypotheses with observed values vs. predicted thresholds |

### Acceptance Criteria
- H6: Glacier CF-faith on non-monotonic config is statistically significantly lower than on base config (Wilcoxon p < 0.05).
- H7: CITRIS Shift-VR degradation is < 10 percentage points across regime configs; CoMTE degradation is > 20 percentage points.
- H4: Spearman ρ < 0.5 for at least three traditional–causal metric pairs.
- All result files are reproducible from fixed seeds documented in `configs/`.

---

## Dependency Graph (Critical Path)

```
Week 1 (DAG + Operators + TSCM)
  └─ Week 2 (Abduction + Action + Prediction)
       └─ Week 3 (LinearSCM-T)
            └─ Week 4 (NlinearSCM-T base)
                 ├─ Week 5 (Classifiers) ──────────────────────┐
                 └─ Week 6 (Ablations + SepsisSim skeleton)    │
                      └─ Week 7 (CF Method Wrappers) ─────────┤
                           └─ Week 8 (Attribution/Concept/     │
                                      Causal Wrappers)         │
                                ├─ Week 9  (Axis A + B) ◄──────┤
                                └─ Week 10 (Axis C + D) ◄──────┘
                                     └─ Week 11 (H1–H5 + Figures)
                                          └─ Week 12 (H6–H7 + Final)
```

---

## Environment and Tooling

```
Python >= 3.11
torch >= 2.3
numpy >= 1.26
scipy >= 1.12
scikit-learn >= 1.4
tslearn >= 0.6        # DTW
torchdiffeq >= 0.2    # Neural-ODE for SepsisSim
pyyaml >= 6.0
pytest >= 8.0
```

Install:
```bash
pip install -e causal_tscf_bench/[dev]
```

---

## Open Risks and Mitigations

| Risk | Week detected | Mitigation |
|---|---|---|
| MIMIC-IV access not granted | 6 | Use synthetic SepsisSim with published clinical DAG from Johnson et al. 2023 |
| Glacier/CITRIS training instability | 7–8 | Reduce learning rate; add gradient clipping; cap training at 100 epochs with early stopping |
| CF-faith ε threshold too strict (H1 not confirmed) | 11 | Re-set ε at 25th percentile instead of 10th; report sensitivity |
| Regime-switching HMM produces undetectable breaks | 6 | Enforce minimum regime duration = 15 timesteps and weight delta ≥ 0.5 |
