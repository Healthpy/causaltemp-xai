# Motivation

No existing benchmark jointly provides ground-truth causal variables, a ground-truth lagged causal graph, and ground-truth structural counterfactuals in a temporal classification setting --- so we cannot currently measure whether temporal XAI methods respect causal structure.

# Gap in the Literature

- **Temporal XAI benchmarks** (CFTS, XTSC-Bench, AMEE, CARLA): measure validity, proximity, sparsity, OOD-plausibility --- but not causal faithfulness.
- **Temporal causal discovery benchmarks** (CausalDynamics, CausalTime, CITRIS benchmarks): measure graph recovery --- but no XAI method or classifier in the loop.
- This paper bridges the two via a four-axis protocol grounded in identifiability theory, explicitly addressing dynamic feedback loops and temporal plausibility constraints.

# Contributions

1. A four-axis evaluation protocol (concept quality, graph quality, counterfactual quality, robustness) with new causal metrics including **ICC**, **CF-faith**, **TRSI**, and **Irreversibility-Violation-Rate**.
2. Two synthetic benchmarks with ground-truth SCMs: **LinearSCM-T**, **NlinearSCM-T** (including non-monotonic and regime-switching ablations).
3. One semi-synthetic clinical benchmark: **SepsisSim** featuring time-varying treatments and dynamic confounding (clinician-validated DAG).
4. Empirical evaluation of methods spanning all four architectural paradigms of temporal CF explainers, showing systematic failure on causal axes.
5. Demonstration that rankings by traditional metrics don't correlate with causal faithfulness (key finding).

---

# Benchmark Construction

All three benchmarks are derived via a unified four-phase generative pipeline grounded in the Temporal Structural Causal Model (TSCM) framework. This section specifies the pipeline so that benchmarks are fully reproducible.

## Phase 1 — Synthetic TSCM Prior Generation

A time-lagged causal DAG $G$ is sampled over $M$ channels with maximum lag $\tau_{\max}$. The structural assignment for each channel $j$ at time $t$ is:

$$X_t^{(j)} := \sum_{X_{t-\tau}^{(i)} \in pa(X_t^{(j)})} w_{ij} \cdot \phi_{ij}\!\left(X_{t-\tau}^{(i)}\right) + U_t^{(j)} \tag{1}$$

The operator $\phi_{ij}$ is sampled uniformly from the **invertible operator dictionary**:

$$\Phi = \{\text{identity},\ \sin,\ \cos,\ \tanh,\ |\cdot|,\ (\cdot)^2,\ \exp(-|\cdot|)\}$$

Coefficients are drawn as $w_{ij} \sim \mathcal{N}(0, \sigma^2_w)$. Exogenous noise $U_t^{(j)}$ is sampled from independent **non-Gaussian** distributions (Laplace or uniform) to ensure identifiability of the causal structure under the Darmois--Skitovic theorem and its non-linear extensions. A Gaussian-noise ablation is reserved as a negative control for Hypothesis 5.

For the regime-switching sub-configuration of NlinearSCM-T, a hidden Markov process with $R \in \{2, 3\}$ regimes triggers structural breaks that modify the active weights $w_{ij}$ at random change-points, introducing controlled non-stationarity.

## Phase 2 — Label Generation via Latent Mechanisms (Label Visibility)

A categorical label $Y \in \{0, 1\}$ is assigned per the **Label Visibility** principle: the class must arise from a threshold crossing of a designated causal variable $j^*$ within the SCM, with downstream effects propagating through all observable channels.

$$Y = \mathbb{1}\!\left[X_T^{(j^*)} > \theta\right]$$

The threshold $\theta$ is tuned per dataset configuration to maintain class balance in $[0.3, 0.7]$. Because the label is a function of the causal variable rather than an independent annotation, any successful structural counterfactual that flips $Y$ must propagate through the SCM — ensuring classification validity is causally grounded, not spurious.

## Phase 3 — Analytical Counterfactual Derivation

For any instance $X$ and a user-specified intervention $do(X_{\mathcal{T}_{int}}^{(i)} = x'_{int})$, the ground-truth counterfactual $X'_{CF}$ is derived via the three-step causal ladder:

**Abduction.** Invert Eq. 1 to recover the latent exogenous noise for every channel and time step:

$$U_t^{(j)} = X_t^{(j)} - \sum_{X_{t-\tau}^{(i)} \in pa(X_t^{(j)})} w_{ij} \cdot \phi_{ij}\!\left(X_{t-\tau}^{(i)}\right) \tag{2}$$

Because $\Phi$ consists of analytically invertible operators and the model is additive, this inversion is exact --- no approximation or normalizing flow is required.

**Action.** Apply the do-operator: replace the structural equation of $X_{\mathcal{T}_{int}}^{(i)}$ with the constant $x'_{int}$, severing all incoming causal edges to that node at the intervention time.

**Prediction.** Forward-simulate the modified SCM using the abduced $U$ and Eq. 1. The result is the unique analytical counterfactual trajectory $X'_{CF}$, which serves as the absolute evaluation target for CF-faith.

## Phase 4 — Unified Evaluation

The generated datasets are handed to fixed black-box classifiers (TCN, LSTM, Transformer trained to $>90\%$ accuracy). Post-hoc explainers generate candidate counterfactuals $X'_{exp}$, which are scored across all four axes.

---

# Four-Axis Evaluation Protocol

## Axis A — Concept Quality

**ICC** (novel) --- Interventional Concept Consistency:

$$\mathrm{ICC}_i = \frac{1}{N} \sum_{n} \mathbb{1}\!\left[ f\!\left(D_{\psi}\!\left(z^{n} + \delta_i e_i\right)\right) \neq f(x^{n}) \right]$$

Under Hyvärinen 2019 / Song 2024 identifiability conditions, high $\mathrm{ICC}$ means causal relevance up to permutation and element-wise reparameterization.

Other metrics:
- **Latent Disentanglement / Entropy Penalization** (new) --- measures if latent exogenous noise variables are properly disentangled from endogenous treatment mechanisms (critical for autoencoder-based counterfactuals).
- **MCC** --- mean correlation coefficient (standard in the identifiability literature).

## Axis B — Temporal Causal Graph Quality

- **SHD** --- Structural Hamming Distance to true lagged adjacency.
- **Lag accuracy** --- fraction of detected edges $(i,j)$ with correct $\tau$.
- **AUC-ROC** --- ranking of edge probability scores.
- **Time-Varying Confounding Robustness** (new) --- evaluates false attribution rates when true drivers are time-varying hidden policies or continuous feedback loops.

**Graph-error decomposition for self-graphing methods.** For methods that infer their own causal graph (CITRIS, iCITRIS), Axis B scores are computed on the inferred graph. CF-faith for these methods is reported in two variants: (i) against the analytical $X'_{CF}$ from the ground-truth graph, and (ii) against a CF derived from their inferred graph. This decomposition isolates how much of the CF-faith gap is attributable to graph error versus propagation failure.

## Axis C — Counterfactual Quality

- **Validity** --- $\mathbb{1}[f(X'_{exp}) = y^*]$.
- **Proximity** --- $L_1 / L_2$ distance $\lVert X'_{exp} - X \rVert_{p}$.
- **Sparsity** --- $L_0$ fraction of altered features.
- **OOD plausibility** --- Isolation Forest (IF), Local Outlier Factor (LOF).
- **TRSI** (Temporal Relevance Smoothness Index) --- measures step-wise temporal coherence; penalizes unfeasible discontinuities between consecutive time steps that arise when perturbations are applied independently without causal propagation. Directly operationalizes Normative Principle P2.
- **CF-faith** (novel) --- normalized DTW distance between the explainer's output and the analytical ground-truth counterfactual:

$$\mathrm{CF\text{-}faith}(X'_{exp}) = 1 - \frac{\mathrm{DTW}(X'_{exp},\ X'_{CF})}{\mathrm{DTW}(X,\ X'_{CF})}$$

  A score of 1 means the explainer exactly recovers the structural counterfactual; 0 means the explainer's output is no closer to $X'_{CF}$ than the original instance. Hard variant: $\mathbb{1}[\mathrm{DTW}(X'_{exp}, X'_{CF}) < \epsilon]$ for a threshold $\epsilon$ set at the 10th percentile of $\mathrm{DTW}(X, X'_{CF})$ per benchmark.

- **Irreversibility-Violation-Rate** (new) --- explicitly penalizes the "Time Traveler Dilemma":

$$\mathrm{IVR} = \frac{1}{N}\sum_{n} \mathbb{1}\!\left[\text{CF sequence alters an irreversible past state or immutable baseline}\right]$$

## Axis D — Robustness

- **Shift-VR** --- validity retention on held-out environments $\mathcal{E}_{\text{test}}$, including regime-switched test splits from NlinearSCM-T.
- **Input sensitivity** --- Lipschitz constant of attribution $\phi_{i,t}$ w.r.t. input.
- **Concept stability** --- Jaccard similarity of top-$k$ concepts across $R$ retraining restarts.

---

# Benchmarks

1. **LinearSCM-T** --- VAR($L$) synthetic with additive linear mechanisms ($\phi_{ij} = \text{identity}$). $k \in \{5, 10\}$ channels, $L = 1$ lag, edge density $s / k^{2} \in \{0.1, 0.2\}$, non-Gaussian noise (Laplace, uniform; Gaussian ablation included as negative control), $T \in \{50, 100\}$, $N = 10{,}000$ sequences. Ships with true graph, mixing matrix, analytically derived CF trajectories (via Eqs. 1--2 + causal ladder), and threshold-based class labels.

2. **NlinearSCM-T** --- Identical DAG sampling, but mechanisms drawn from the invertible operator dictionary $\Phi$ (Eq. 1), enabling exact abduction via Eq. 2. Sub-configurations:
   - **Base**: operators from $\Phi$ excluding step functions.
   - **Non-monotonic ablation**: adds discrete step functions $\phi_{ij} = \mathbb{1}[\cdot > 0]$ to test conditions where standard flow-based identifiability (e.g., DoFlow) breaks down.
   - **Regime-switching ablation**: HMM with $R \in \{2, 3\}$ regimes modifies active $w_{ij}$ at structural break-points; used to evaluate Shift-VR under temporal non-stationarity.

3. **SepsisSim** --- Real-data grounded, clinician-validated DAG on MIMIC-IV sepsis cohort. Incorporates time-varying treatments (e.g., dynamic fluid administration) and dynamic confounding to stress-test sequence-to-sequence causal mechanisms. Labels derived from clinically meaningful outcome thresholds (e.g., MAP crossing 65 mmHg). Uses Neural-ODE transitions for continuous-time state evolution within each regime.

---

# Methods Evaluated

Methods are selected to cover all four architectural paradigms of temporal CF explainers, so that paradigm-level failure modes are isolated rather than conflated.

| Paradigm | Family (plan label) | Methods | Characteristic failure mode |
| :--- | :--- | :--- | :--- |
| Instance substitution | Counterfactual | **CoMTE** | SCM consistency violation: donor segment incompatible with recipient's $U$ |
| Evolutionary / heuristic | Counterfactual | **TSEvo** | Independent mutation of parent/child variables; breaks lead-lag structure |
| Deep latent / generative | Counterfactual | **Glacier** | Reverts to associative correlations under out-of-distribution do-interventions |
| Amortized / controllable | Causal baseline | **CARLA-causal** | Learned masks unvalidated for causal faithfulness; severs real dependencies |
| Attribution | Attribution | **TimeSHAP**, **Dynamask** | No CF generation; evaluated on Axes A and D only |
| Concept-based | Concept-based | **CBM-T**, **iVAE** | Evaluated on Axis A; iVAE additionally on Axis B via its inferred graph |
| Causal / graph-aware | Causal baseline | **CITRIS**, **iCITRIS** | Evaluated on Axes B + C with graph-error decomposition |
| Naive baseline | Counterfactual | **DiCE** | Tabular-style perturbation; no temporal structure |

**Classifiers held fixed:** TCN, LSTM, Transformer --- all trained to $>90\%$ accuracy per benchmark before any explainer is run.

---

# Normative Design Principles the Benchmark Operationalizes

| Principle | What it requires | Measured by |
| :--- | :--- | :--- |
| P1: Granularity alignment | Interventions only at valid control points | Tier 1 validity; action-space sparsity |
| P2: Dynamical propagation | Changes propagate via structural equations (Eq. 1) | CF-faith; TRSI |
| P3: Constraint preservation | Domain constraints in generative process | OOD plausibility + IVR |
| P4: Policy-level framing | CF expressed as action sequences, not feature perturbations | Action-decoding success rate on SepsisSim |
| P5: Uncertainty quantification | CFs include uncertainty bounds | Bootstrap CIs on all metrics; Axis D robustness |
| P6: Label Visibility | Class label arises from a causal threshold crossing; its effects propagate through all observable channels | Verified by construction; ablation: decorrelated label as null control |

---

# Hypotheses

H1. Standard temporal XAI methods achieve low CF-faith ($< 0.3$) despite high validity ($> 0.9$).

H2. Standard methods fail to differentiate causal from non-causal concepts via $\mathrm{ICC}$.

H3. Causal-recourse and CITRIS-based methods achieve CF-faith $> 0.7$ with moderate validity/proximity degradation.

H4. *(Key finding)* Rankings by traditional metrics don't correlate with rankings by CF-faith or $\mathrm{ICC}$ (Spearman $\rho < 0.5$).

H5. Identifiability-condition satisfaction predicts empirical CF-faith: methods evaluated on non-Gaussian NlinearSCM-T configurations outperform those on the Gaussian ablation; performance degrades monotonically as graph sparsity decreases below $s/k^2 = 0.1$.

H6. Generative flow models (DoFlow, Glacier) fail to recover true structural counterfactuals under the non-monotonic NlinearSCM-T ablation, while additive-mechanism methods remain unaffected.

H7. Under regime-switching NlinearSCM-T, Shift-VR degrades faster for instance-substitution methods (CoMTE) than for causal-graph-aware methods (CITRIS), reflecting their structural incompatibility with distributional breaks.

---

# Scope Boundaries: Open Problems the Benchmark Does NOT Address

- Synthetic benchmarks assume causal sufficiency; SepsisSim introduces dynamic confounding, but exhaustive hidden confounder discovery is left for future work.
- NlinearSCM-T exercises nonlinearity but without tight identifiability guarantees for arbitrary non-monotone functions (step-function ablation is intentionally adversarial, not identifiable).
- We do not benchmark joint-training efficiency of causal methods.
- $\mathrm{ICC}$ uses synthetic ground-truth concepts; no user study on SepsisSim.
- SepsisSim uses Neural-ODE transitions but not hybrid-mode transitions; continuous-time benchmark deferred.
- Topological Causality (TC) as a phase-space causal validity metric is not included in the main evaluation; its $O(T \cdot d \cdot \log T + T \cdot k \cdot d^3)$ cost renders it infeasible at the benchmark scale and is deferred to a targeted NlinearSCM-T sub-study.
