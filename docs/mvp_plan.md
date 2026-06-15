# CausalTemp-XAI v0.1

## MVP Research Plan

*4-week sprint · 3 authors · one synthetic benchmark · causal faithfulness as the key signal*

## MVP GOAL

Deliver a minimal, fully reproducible benchmark that demonstrates the key phenomenon: traditional temporal XAI metrics can disagree with causal metrics (CF-faith / ICC) on at least one synthetic SCM benchmark and one family of methods.

## 1. MVP Scope

The MVP deliberately narrows scope to what is necessary and sufficient to establish the key phenomenon. Everything excluded is documented as future work toward the full v1.0 paper.

| Dimension | ✓ In Scope (MVP) | ✗ Out of Scope (future work) |
|---|---|---|
| Benchmarks | LinearSCM-T — single fixed configuration (k, L, sparsity, noise, T, N). Optional: 1 extra config varying sparsity to hint at H5. | NlinearSCM-T. SepsisSim and all real / semi-synthetic datasets. |
| Axes | Axis C (full): validity, proximity, sparsity, OOD plausibility, CF-faith hard + soft. Axis D (partial): Shift-VR-lite. | Axis A (ICC, MIG, DCI, MCC). Axis B (graph recovery metrics SHD, lag accuracy, AUC-ROC). |
| Methods | 1–2 attribution baselines (Integrated Gradients, TimeSHAP). 2–3 CF methods: Wachter, DiCE/SETS, CARLA-causal recourse baseline. | Full concept-based zoo (TCAV-T, CBMT, LEAP, iVAE, β-VAE). RL-CF. Complex causal discovery integration. |
| Classifiers | Single TCN or LSTM architecture tuned to >90% test accuracy. | Transformer variants. |

## 2. Author Roles

How the roles might be splitted:

| Author | Hat | Responsibilities |
|---|---|---|
| Author A | Causal / Theory | Design LinearSCM-T generator (single config). Define CF-faith (hard + soft) and Shift-VR-lite. Write method-agnostic benchmark and evaluation sections of the paper. |
| Author B | Systems / Experiments | Implement generator + data splits. Train TCN to >90% accuracy. Integrate 1–2 attribution and 2–3 CF methods. Run all experiments; produce plots and tables. |
| Author C | Writing / Positioning | Write intro, related work (temporal XAI + causal discovery gap), and discussion. Shape narrative around the key empirical finding: validity rankings vs. CF-faith rankings diverge. |

## 3. Work Packages

### WP1 — Minimal Benchmark + Classifier

**LinearSCM-T single configuration**

- Fix one canonical configuration: e.g., k=10 latents, L=1 lag, sparsity s/k²=0.2, Laplace noise, T=100, N=10,000 sequences.
- Export three artifacts per split: time-series X, labels Y, and the ground-truth SCM (lagged adjacency + mechanism parameters) required for CF-faith checking.
- Implement unit tests on the generator: verify sparsity matches target, noise distribution passes KS test, causal ordering is acyclic.
- Optional (time permitting): generate one additional config varying sparsity to s/k²=0.1 as a preliminary H5 data point.

**Classifier**

- 60/20/20 train/val/test split. Implement early stopping on val loss (patience=10).
- Target: >90% test accuracy. If not reached within a reasonable architecture search, document the gap and continue — benchmark evaluation does not depend on classifier perfection.
- Freeze and checkpoint weights before any XAI evaluation begins.

### WP2 — Minimal CF Evaluation Pipeline

**Counterfactual generators (2–3)**

- Wachter-style optimization: gradient-based perturbation minimizing distance subject to label flip. Implement from scratch or adapt CARLA.
- Off-the-shelf library method: DiCE or SETS — use official implementation to minimize re-implementation risk.
- Causal recourse baseline: CARLA-causal (Karimi et al.) — the key comparison point that should show higher CF-faith.

**Axis C metrics**

- Validity: fraction of CFs that flip the classifier label.
- Proximity: L1 and L2 distance between CF and original time series.
- Sparsity: L0 fraction of altered features across the time axis.
- OOD plausibility: Isolation Forest (IF) and Local Outlier Factor (LOF) scores computed on training set, evaluated on generated CFs.
- CF-faith (hard): binary indicator. Simulate SCM under proposed intervention; check (i) no retroactive change in pre-intervention timesteps, and (ii) forward propagation matches structural equations for a fixed number of steps post-intervention.
- CF-faith (soft): continuous relaxation of both conditions via L1 residual from SCM simulation.

**Axis D — Shift-VR-lite**

- Generate a second environment by modifying noise distribution (e.g., swap Laplace for Uniform) or adding mild covariate shift to input marginals.
- Re-evaluate validity and CF-faith on this shifted environment. Report Shift-VR = fraction of originally-valid CFs that remain valid.

### WP3 — Attribution Baseline

- Implement Integrated Gradients (and optionally TimeSHAP) against the TCN.
- Use attributions to compute a saliency-based validity proxy: deletion/insertion curves (sequentially mask high-attribution timesteps and measure accuracy drop). Keep this lightweight — the goal is a foil, not a full attribution benchmark.
- Key demonstration: show that high attribution-based saliency quality coexists with low CF-faith for the same model and data, establishing the central empirical gap.

### WP4 — Analysis & Narrative

**Three core figures**

- Figure 1 — Validity vs CF-faith scatter: one point per method per evaluation instance. This is the key figure; methods should cluster visibly by family.
- Figure 2 — CF-faith distribution: violin or histogram of CF-faith (hard and soft) across all methods, showing systematic failure of standard CF methods.
- Figure 3 — Rank correlation: Spearman ρ between each traditional metric (validity, proximity, sparsity) and CF-faith, even in reduced form over the 3–4 methods evaluated.

**Paper skeleton (5–8 pages)**

- Abstract + Introduction: state the gap, the MVP benchmark, and the key finding.
- Background: two-cluster related work (temporal XAI benchmarks; temporal causal discovery benchmarks).
- Evaluation Protocol: CF-faith definition, Axis C metrics, Shift-VR-lite. Note Axes A/B as future work.
- LinearSCM-T: dataset description, SCM parameterization, classifier training.
- Experiments: results tables + three core figures.
- Discussion: interpret rank divergence; enumerate limitations and the path to v1.0.

## 4. 4-Week Sprint Timeline

| Week | Focus | Concrete Milestones | Owner |
|---|---|---|---|
| Week 1 | Benchmark generation + classifier pipeline | LinearSCM-T config locked. Generator passes unit tests. TCN training pipeline runs end-to-end. | A (config), B (code) |
| Week 2 | TCN converges + first CF method + CF-faith | TCN >90% test accuracy. Wachter integrated. CF-faith (hard + soft) checker implemented and validated against manual SCM simulation. | B (TCN, Wachter), A (CF-faith) |
| Week 3 | Remaining methods + full metric sweep | DiCE/SETS and CARLA-causal integrated. IG attribution baseline done. All Axis C + Shift-VR-lite metrics computed on test set and shifted environment. | B (integration), A (metrics review) |
| Week 4 | Analysis, figures, paper draft + go/no-go | Three core figures finalized. Tables formatted. 5–8 page draft complete. Go/no-go decision: extend to more configs/methods (v1.0) or freeze as v0.1 benchmark. | C (writing), A+B (review) |

**END-OF-SPRINT DECISION**

At Week 4, decide: (a) freeze v0.1 as a workshop paper / preprint, or (b) extend to NlinearSCM-T + SepsisSim + full method zoo toward a full conference submission. The MVP must stand on its own regardless of which path is chosen.

## 5. Hypotheses Testable in the MVP

The four-week sprint can partially test three of the five full-paper hypotheses. Pre-register these before running experiments.

| Hypothesis | MVP Status | Operationalization in MVP | Expected Result |
|---|---|---|---|
| H1 | ✓ Testable | Standard CF methods (Wachter, DiCE) achieve validity >0.9 on LinearSCM-T single config; CF-faith (hard) <0.3. | Confirmed |
| H3 | ✓ Testable | CARLA-causal achieves CF-faith >0.7 with moderate validity/proximity degradation vs. Wachter. | Confirmed |
| H4 | ✓ Partial | Spearman ρ between traditional metrics (validity, proximity) and CF-faith across 3–4 methods. Small N — treat as preliminary evidence, not definitive claim. | ρ < 0.5 |
| H2 | ✗ Deferred | Requires Axis A (ICC) — deferred to v1.0. | — |
| H5 | ◑ Hint only | If optional second sparsity config is generated, compare CF-faith across the two. No ablation power — descriptive only. | Descriptive |

## 6. MVP Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| TCN does not reach >90% accuracy in time | Medium | Relax target to >85% and document. CF-faith is classifier-agnostic — the phenomenon persists at lower accuracy. Allocate buffer at end of Week 1. |
| CARLA-causal or DiCE integration breaks on time-series input | Medium | Pre-check library compatibility in Week 1. If integration fails, implement a minimal gradient-based causal recourse proxy instead — document the deviation. |
| H1 not confirmed (standard CF methods achieve high CF-faith too) | Low | Still a publishable finding — report and analyze why. Pre-registration protects against HARKing. Revisit SCM parameterization for causal sufficiency. |
| Spearman ρ (H4 partial) is not below 0.5 with only 3–4 methods | Medium | Flag as underpowered; do not claim H4 confirmed in MVP paper. Extend method count in v1.0. The scatter plot (Figure 1) is sufficient visual evidence for the workshop version. |
| 4 weeks is too short to write a clean 6–8 page draft | Medium | Author C starts writing related work and protocol sections in parallel with WP1 (Week 1). Figures can be placeholder until Week 4. |

## 7. Definition of Done — v0.1

The MVP sprint is complete when all of the following are true:

- ☐ LinearSCM-T single config generated, tested, and documented.
- ☐ TCN checkpoint frozen with test accuracy logged.
- ☐ Wachter, DiCE/SETS, and CARLA-causal all produce valid output on LinearSCM-T.
- ☐ CF-faith (hard + soft) checker validated against manual SCM simulation on ≥10 examples.
- ☐ All Axis C metrics + Shift-VR-lite computed and stored in a reproducible results file.
- ☐ Three core figures are publication-quality (not placeholder).
- ☐ H1 and H3 formally assessed with point estimates (no CI required for MVP).
- ☐ 6–8 page draft exists with all sections present (even if some are stubs).
- ☐ All code committed to a shared repository with a README that reproduces results end-to-end.
- ☐ Go/no-go decision documented with rationale.
