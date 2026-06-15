# Motivation

No existing benchmark jointly provides ground-truth causal variables, a
ground-truth lagged causal graph, and ground-truth structural
counterfactuals in a temporal classification setting --- so we cannot
currently measure whether temporal XAI methods respect causal structure.

# Gap in the Literature

-   **Temporal XAI benchmarks** (CFTS, XTSC-Bench, AMEE, CARLA): measure
    validity, proximity, sparsity, OOD-plausibility --- but not causal
    faithfulness.

-   **Temporal causal discovery benchmarks** (CausalDynamics,
    CausalTime, CITRIS benchmarks): measure graph recovery --- but no
    XAI method or classifier in the loop.

-   This paper bridges the two via a four-axis protocol grounded in
    identifiability theory.

# Contributions

1.  A four-axis evaluation protocol (concept quality, graph quality,
    counterfactual quality, robustness) with three new metrics: **ICC**,
    **CF-faith**, **T-stab**.

2.  Two synthetic benchmarks with ground-truth SCMs: **LinearSCM-T**,
    **NlinearSCM-T**.

3.  One semi-synthetic clinical benchmark: **SepsisSim**
    (clinician-validated DAG).

4.  Empirical evaluation of 10+ XAI methods showing systematic failure
    on causal axes.

5.  Demonstration that rankings by traditional metrics don't correlate
    with causal faithfulness (key finding).

# Four-Axis Evaluation Protocol

## Axis A --- Concept Quality

**ICC** (novel) --- Interventional Concept Consistency:
$$\mathrm{ICC}_i \;=\; \frac{1}{N} \sum_{n} \mathbb{1}\!\left[
    f\!\left(D_{\psi}\!\left(z^{n} + \delta_i e_i\right)\right) \neq f(x^{n})
  \right].$$ Under Hyvärinen 2019 / Song 2024 identifiability
conditions, high $\mathrm{ICC}$ means causal relevance up to permutation
and element-wise reparameterization.

Other metrics:

-   **MIG** (Chen et al. 2018) --- mutual information gap.

-   **DCI-D, DCI-C** (Eastwood & Williams 2018) --- disentanglement,
    completeness.

-   **MCC** --- mean correlation coefficient (standard in the
    identifiability literature).

## Axis B --- Temporal Causal Graph Quality

-   **SHD** --- Structural Hamming Distance to true lagged adjacency.

-   **Lag accuracy** --- fraction of detected edges $(i,j)$ with correct
    $\tau$.

-   **AUC-ROC** --- ranking of edge probability scores.

## Axis C --- Counterfactual Quality

-   **Validity**.

-   **Proximity** --- $L_1 / L_2$ distance
    $\lVert \hat{x}^{n} - x^{n} \rVert_{p}$.

-   **Sparsity** --- $L_0$ fraction of altered features.

-   **OOD plausibility** --- IF, LOF.

-   **CF-faith** (novel) --- indicator: no-retroactive-change $\wedge$
    graph-propagation holds. Hard and soft variants.

-   **T-stab** (novel) --- trajectory coherence across related inputs.

## Axis D --- Robustness

-   **Shift-VR** --- validity retention on held-out environments
    $\mathcal{E}_{\text{test}}$.

-   **Input sensitivity** --- Lipschitz constant of attribution
    $\phi_{i,t}$ w.r.t. input.

-   **Concept stability** --- Jaccard similarity of top-$k$ concepts
    across $R$ retraining restarts.

# Benchmarks

1.  **LinearSCM-T** --- VAR($L$) synthetic. $k \in \{5, 10, 20\}$
    latents, $L \in \{1, 3\}$ lags, sparsity
    $s / k^{2} \in \{0.1, 0.2, 0.3\}$, non-Gaussian noise (Laplace,
    uniform), $S \in \{5, 10\}$ segments (for (B2) segment variability),
    $T \in \{50, 100, 200\}$, $N = 10{,}000$ sequences. Ships with true
    graph, mixing, CF trajectories.

2.  **NlinearSCM-T** --- Same structure, MLP mechanisms and nonlinear
    mixing. Tests generalization beyond linear VAR identifiability.

3.  **Movement SCMs** that you've created to validate your method.

**May be reused:** CausalDynamics (Axis B reference), Temporal
Causal3DIdent (Axis A reference), MIMIC-IV sepsis cohort (real-data
sanity check), UCR/UEA subset (breadth, Axis C+D only).

# Methods Evaluated

  **Family**         **Methods**
  ------------------ --------------------------------------------------------------------------------------------------------------------------------------
  Attribution        KernelSHAP, Integrated Gradients, TimeSHAP, Dynamask, Attribution Stability Indicator
  Concept-based      TCAV-T, CBM-T, LEAP, iVAE, $\beta$-VAE
  Counterfactual     Wachter, DiCE, Native Guide, COMTE, SETS, TSEvo, Glacier, M-CELS, Sub-SpaCE, MASCOTS, CONFETTI, RL-CF
  Causal baselines   Karimi Causal Recourse (CARLA-causal), CITRIS, iCITRIS, CtrlNS-encoder + do-calculus CF, DYNOTEARS/PCMCI-informed post-hoc filtering

**Classifiers held fixed:** TCN, LSTM, Transformer --- all trained to
$>90\%$ accuracy per benchmark.

# Normative Design Principles the Benchmark Operationalizes

  **Principle**                    **What it requires**                                          **Measured by**
  -------------------------------- ------------------------------------------------------------- -------------------------------------------------
  P1: Granularity alignment        Interventions only at valid control points                    Tier 1 validity; action-space sparsity
  P2: Dynamical propagation        Changes propagate via structural equations                    CF-faith (soft); dynamics-residual
  P3: Constraint preservation      Domain constraints in generative process                      OOD plausibility + domain-specific range checks
  P4: Policy-level framing         CF expressed as action sequences, not feature perturbations   Action-decoding success rate on SepsisSim
  P5: Uncertainty quantification   CFs include uncertainty bounds                                Bootstrap CIs on all metrics; Axis D robustness

# Hypotheses

::: description
Standard temporal XAI methods achieve low CF-faith ($< 0.3$) despite
high validity ($> 0.9$).

Standard methods fail to differentiate causal from non-causal concepts
via $\mathrm{ICC}$.

Causal-recourse and CITRIS-based methods achieve CF-faith $> 0.7$ with
moderate validity/proximity degradation.

*(Key finding)* Rankings by traditional metrics don't correlate with
rankings by CF-faith or $\mathrm{ICC}$ (Spearman $\rho < 0.5$).

Identifiability-condition satisfaction predicts empirical performance
(ablations on sparsity, segment count, Gaussianity).
:::

# Scope Boundaries: Open Problems the Benchmark Does NOT Address

::: description
Benchmarks assume causal sufficiency; unmeasured confounders left for
future work.

NlinearSCM-T exercises nonlinearity but without tight identifiability
guarantees.

We don't benchmark joint-training efficiency of causal methods.

$\mathrm{ICC}$ uses synthetic ground-truth concepts; no user study on
SepsisSim.

SepsisSim uses Neural-ODE transitions but not hybrid-mode transitions;
continuous-time benchmark deferred.
:::
