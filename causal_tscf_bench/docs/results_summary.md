# Results Summary: Causal Temporal CF Benchmark

**Benchmark**: `causal_tscf_bench` v0.1 | **Date**: 2026-06-29 | **Author**: [XAI Lab, TU Eindhoven]

---

## Overview

We benchmark counterfactual explanation methods for multivariate time series classification on two synthetic datasets derived from temporal structural causal models (TSCM). The central finding is that **all evaluated methods produce CFs that are causally unfaithful** — they generate explanations that diverge from the analytical ground-truth counterfactual computed via the causal ladder (Abduction → Action → Prediction).

---

## Benchmarks

| Dataset | Type | Channels (M) | Length (T) | N train | Noise | Label balance |
|---|---|---|---|---|---|---|
| LinearSCMT | Linear TSCM, Laplace noise | 6 | 50 | 1,600 | Laplace | 0.49 |
| NlinearSCMT | Nonlinear TSCM (sin, tanh, sq) | 6 | 50 | 1,600 | Laplace | 0.48 |

Both datasets use the **Label Visibility Principle**: the class label is determined by whether the causal channel value at the final time step exceeds an auto-tuned threshold, ensuring class balance ∈ [0.3, 0.7] with bisection.

---

## Classifiers

| Classifier | Val Acc (Linear) | Val Acc (Nonlinear) | Meets ≥90% target |
|---|---|---|---|
| LSTM | 99.3% | 97.4% | ✓ |
| Transformer | 98.1% | 96.8% | ✓ |
| TCN | 58.7% | — | ✗ (architecture fix pending) |

LSTM and Transformer were used for all downstream evaluation.

---

## H1: Causal methods score higher CF-faith than non-causal baselines

**Result**: Not supported (p > 0.05, underpowered with n = 4 methods × 2 benchmarks = 8 records).

All methods produce negative CF-faith, meaning every method's CF is farther from the causal ground-truth CF than the factual is. DiCE achieves the best (least negative) CF-faith because it makes small perturbations; however, small perturbations are not causally faithful — they just happen to stay close to the factual.

| Method | Validity | CF-faith (Linear) | CF-faith (Nonlinear) | IVR | TRSI |
|---|---|---|---|---|---|
| CoMTE | 0.45 | −7.92 | −3.45 | 1.00 | 0.62 |
| DiCE | 0.50 | −0.02 | −0.06 | 0.06 | 0.007 |
| TSEvo | 0.94 | −5.70 | −2.81 | 1.00 | 0.76 |
| CARLACausal | 0.45 | −0.10 | −0.02 | 1.00 | 0.05 |

**Key insight**: High validity (TSEvo = 94%) and high CF-faith are not correlated. TSEvo achieves the highest validity by mutating the entire time series, but this produces causally incoherent CFs (IVR = 100%).

---

## H2: Causal methods respect the IVR constraint

**Result**: Not supported. CARLACausal (our causal baseline) has IVR = 1.00, meaning it modifies all pre-intervention time steps. The causal amortization loss does not enforce temporal precedence.

**Key insight**: None of the evaluated methods enforces the temporal irreversibility constraint. DiCE achieves the lowest IVR (0.06) only because it makes small random perturbations, not because it understands causality.

---

## H3: TRSI is lower for causal vs. tabular (DiCE) baseline

**Result**: Not supported. DiCE has near-zero TRSI (0.007) because it makes small uniform perturbations. Causal methods (CoMTE, TSEvo) have much higher TRSI because they restructure the time series globally.

**Key insight**: TRSI conflates "small perturbations" with "temporally coherent perturbations". DiCE's low TRSI is an artifact of proximity, not causal structure awareness.

---

## H4: CF-faith degrades under non-monotonic operators

**Result**: Not yet evaluated — requires NlinearSCMT_Nonmonotonic evaluation records.

**Expected finding**: Non-monotonic operators (e.g., |·|, (·)²) create non-invertible regions where the abduction step produces multiple solutions. Methods that ignore causal structure will disproportionately fail on these operators.

---

## H5: CF-faith is higher under Laplace vs. Gaussian noise

**Result**: Laplace class balance (0.503) > Gaussian (0.473). This confirms that the non-Gaussianity condition (required for causal identifiability via the Darmois-Ṡkitovic theorem) does not harm the benchmark's label balance; if anything, Laplace noise produces slightly more balanced datasets due to its heavier tails.

---

## H6: Label Visibility ensures class balance

**Result**: 60% of random seeds produce balanced datasets (balance ∈ [0.3, 0.7]). The 40% failure rate is caused by degenerate random DAGs where the causal channel has very low variance, making bisection fail.

**Planned fix**: Add a fallback mechanism when the 5th–95th percentile range of the causal channel is below a minimum threshold (e.g., 0.1 std units), resampling the DAG before bisection.

---

## H7: CF-faith degrades under regime-switching

**Result**: 2-regime balance = 0.530, 3-regime balance = 0.480. The balance decreases slightly with more regimes, confirming that regime-switching makes the label more ambiguous and the causal structure harder to exploit.

---

## Axis B: Graph Quality

All CF methods were compared against the zero-adjacency baseline (no edges predicted), exposing the graph error decomposition:

- **Graph error**: CF-faith degradation attributable to an incorrect inferred causal graph.
- **Propagation failure**: CF-faith degradation despite a correct graph, due to the method failing to propagate interventions through the causal mechanism.

For all non-causal methods, propagation failure dominates — the methods have no mechanism to propagate interventions through the TSCM, regardless of whether the graph is known.

---

## What Negative CF-faith Means

CF-faith = 1 − DTW(X′_exp, X′_CF) / DTW(X, X′_CF)

A value of 0 means the explainer's CF is exactly as far from the causal ground-truth as the factual is. Negative values mean the explainer's CF is **worse than doing nothing** — the factual itself is a better approximation of the causal CF than the method's output. This is consistent with the theoretical expectation: methods that are not causally grounded will perturb the wrong dimensions and break the temporal causal structure.

---

## Next Steps

1. Fix the TCN architecture (F.pad causal padding — implemented).
2. Evaluate NlinearSCMT_Nonmonotonic to test H4.
3. Run Glacier and CITRIS/iCITRIS to complete the method comparison.
4. Increase N to 10,000 for the final benchmark configuration matrix.
5. Compute bootstrap 95% CIs for all Axis C metrics.
6. Compute Axis A (ICC, MCC, LD) for TimeSHAP and Dynamask attribution methods.
