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

A time-lagged causal DAG $G$ is sampled over $M$ channels with maximum lag $\tau_{\max}$. Each channel $j$ at time $t$ follows an **additive-noise structural equation** (ANM) over its masked lagged parents:

$$X_t^{(j)} := f_j\!\left(\mathrm{pa}(X_t^{(j)})\right) + U_t^{(j)} \tag{1}$$

Additive noise is deliberate: it makes Pearl abduction the *exact* subtraction $U = X - f(\mathrm{pa})$, which is what makes CF-faith and the oracle structural counterfactual exact (§Phase 3). Two mechanism families are implemented (the source of truth is the code — see `docs/spec_code_reconciliation.md` §1):

- **LinearSCM-T** — $f_j$ linear (VAR): $X_t^{(j)} = \sum_{X_{t-\tau}^{(i)} \in \mathrm{pa}} w_{ij}\, X_{t-\tau}^{(i)} + U_t^{(j)}$.
- **NlinearSCM-T** — $f_j$ an additive-noise per-node MLP: $X_t^{(j)} = \mathrm{decay}_j X_{t-1}^{(j)} + \mathrm{gain}\cdot\tanh\!\big(\mathrm{MLP}_j(\text{masked lagged parents})\big) + U_t^{(j)}$, with spectral-norm-capped weights for bounded long-horizon dynamics.

Coefficients/weights are drawn from $\mathcal{N}(0, \sigma^2_w)$. Exogenous noise $U_t^{(j)}$ is sampled from independent **non-Gaussian** distributions (Laplace or uniform) for identifiability under Darmois–Skitovic and its non-linear extensions; a Gaussian-noise ablation is the H5 negative control.

> **Operator dictionary (footnote).** An alternative per-edge parameterisation $\phi_{ij}\in\Phi=\{\text{identity},\sin,\cos,\tanh,|\cdot|,(\cdot)^2,\exp(-|\cdot|)\}$ is implemented in `causaltemp_xai/scm/operators.py` but is a **dormant alternative** (not wired into the benchmark pipeline). The **non-monotonic NlinearSCM-T ablation** draws the non-invertible step function $\mathbb{1}[\cdot>0]$ from this pool (H6). The per-node MLP of NlinearSCM-T subsumes a sum of per-edge operators while keeping additive noise.

For the regime-switching sub-configuration of NlinearSCM-T, a hidden Markov process with $R \in \{2, 3\}$ regimes triggers structural breaks that modify the active weights $w_{ij}$ at random change-points, introducing controlled non-stationarity.

## Phase 2 — Label Generation via Latent Mechanisms (Label Visibility)

A categorical label $Y \in \{0, 1\}$ is assigned per the **Label Visibility** principle: the class must arise from a threshold crossing of a designated causal variable $j^*$ within the SCM, with downstream effects propagating through all observable channels.

$$Y = \mathbb{1}\!\left[X_T^{(j^*)} > \theta\right]$$

The threshold $\theta$ is tuned per dataset configuration to maintain class balance in $[0.3, 0.7]$. Because the label is a function of the causal variable rather than an independent annotation, any successful structural counterfactual that flips $Y$ must propagate through the SCM — ensuring classification validity is causally grounded, not spurious.

## Phase 3 — Analytical Counterfactual Derivation

For any instance $X$ and a user-specified intervention $do(X_{\mathcal{T}_{int}}^{(i)} = x'_{int})$, the ground-truth counterfactual $X'_{CF}$ is derived via the three-step causal ladder:

**Abduction.** Invert Eq. 1 to recover the latent exogenous noise for every channel and time step:

$$U_t^{(j)} = X_t^{(j)} - f_j\!\left(\mathrm{pa}(X_t^{(j)})\right) \tag{2}$$

Because the model is **additive-noise**, this inversion is an *exact subtraction* for both implemented families — linear $f_j$ and the additive-noise MLP — with no approximation or normalizing flow required. (In `causaltemp_xai/benchmarks/structural_cf.py::abduct_noise`, $f_j$ is the mechanism's deterministic next-step mean.)

**Action.** Apply the do-operator: replace the structural equation of $X_{\mathcal{T}_{int}}^{(i)}$ with the constant $x'_{int}$, severing all incoming causal edges to that node at the intervention time.

**Prediction.** Forward-simulate the modified SCM using the abduced $U$ and Eq. 1. The result is the unique analytical counterfactual trajectory $X'_{CF}$, which serves as the absolute evaluation target for CF-faith.

## Phase 4 — Unified Evaluation

The generated datasets are handed to fixed black-box classifiers (TCN, LSTM, Transformer trained to $>90\%$ accuracy). Post-hoc explainers generate candidate counterfactuals $X'_{exp}$, which are scored across all four axes.

---

# Four-Axis Evaluation Protocol

## Axis A — Concept Quality

**ICC** (novel) --- Interventional Concept Consistency:

$$\mathrm{ICC}_i = \frac{1}{N} \sum_{n} \mathbb{1}\!\left[ f\!\left(D_{\psi}\!\left(z^{n} + \delta_i e_i\right)\right) \neq f(x^{n}) \right]$$

Under Hyvärinen 2019 / Song 2024 identifiability conditions, high $\mathrm{ICC}$ means causal relevance up to permutation and element-wise reparameterization. This latent-traversal form is implemented as `axis_a.icc_latent` and is the **definitional** ICC; it applies to methods that expose an encoder/decoder $D_\psi$ (concept / representation learners — iVAE, CITRIS, CBM-T). **Attribution methods** (TimeSHAP, Dynamask, IG) have no $D_\psi$, so they are scored with a **decoder-free proxy** (`axis_a.icc`) that measures attribution-mass concentration on the causally-relevant channel. See `docs/spec_code_reconciliation.md` §2; wiring `icc_latent` into a decoder-based experiment is a tracked P1 follow-up.

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
- **CF-faith** (novel) --- **mechanism-residual SCM-consistency** of the explainer's counterfactual (the implemented, canonical form — see `docs/spec_code_reconciliation.md` §3). A CF is faithful iff (i) it makes **no retroactive** pre-intervention change and (ii) its post-intervention trajectory is consistent with re-rolling the *known SCM mechanism* forward from the intervention. Let $r(X'_{exp})$ be the mean L1 residual between $X'_{exp}$ and that mechanism rollout. Then:

$$\mathrm{CF\text{-}faith}_{\text{soft}} = \exp\!\left(-\,r(X'_{exp})/s\right)\in[0,1], \qquad \mathrm{CF\text{-}faith}_{\text{hard}} = \mathbb{1}\!\left[\text{no retroactive change} \ \wedge\ r(X'_{exp}) < \mathrm{tol}\right]$$

  Reported under **two semantics**: *noiseless-rollout* (re-roll with $U=0$; the deterministic skeleton CF) and *pearl-delta* (abduct the factual noise and re-inject it). These are mutually exclusive by construction and expose the rollout-vs-Pearl contrast that is central to the benchmark. *(An alternative — normalized DTW to the analytical oracle CF, $1 - \mathrm{DTW}(X'_{exp},X'_{CF})/\mathrm{DTW}(X,X'_{CF})$ — is cheaply computable since the oracle CF exists, and MAY be added as a complementary secondary metric; it is not the canonical CF-faith.)*

- **Irreversibility-Violation-Rate** (new) --- explicitly penalizes the "Time Traveler Dilemma":

$$\mathrm{IVR} = \frac{1}{N}\sum_{n} \mathbb{1}\!\left[\text{CF sequence alters an irreversible past state or immutable baseline}\right]$$

## Axis D — Robustness

- **Shift-VR** --- validity retention on held-out environments $\mathcal{E}_{\text{test}}$, including regime-switched test splits from NlinearSCM-T.
- **Input sensitivity** --- Lipschitz constant of attribution $\phi_{i,t}$ w.r.t. input.
- **Concept stability** --- Jaccard similarity of top-$k$ concepts across $R$ retraining restarts.

---

# Benchmarks

1. **LinearSCM-T** --- VAR($L$) synthetic with additive linear mechanisms ($\phi_{ij} = \text{identity}$). $k \in \{5, 10\}$ channels, $L = 1$ lag, edge density $s / k^{2} \in \{0.1, 0.2\}$, non-Gaussian noise (Laplace, uniform; Gaussian ablation included as negative control), $T \in \{50, 100\}$, $N = 10{,}000$ sequences. Ships with true graph, mixing matrix, analytically derived CF trajectories (via Eqs. 1--2 + causal ladder), and threshold-based class labels.

2. **NlinearSCM-T** --- Identical DAG sampling, but additive-noise **per-node MLP** mechanisms (Eq. 1, `MLPMechanism`; spectral-norm-capped), keeping exact abduction via Eq. 2. Sub-configurations (as implemented — see `docs/spec_code_reconciliation.md` §1):
   - **Base** (`smoke_nl` / `full_nl`): monotone `tanh` hidden activation.
   - **Non-monotonic ablation** (`smoke_nonmonotonic`, H6): `sin` hidden activation (bounded, 1-Lipschitz, non-monotone) — tests where flow-based identifiability breaks down. *(A true step-function variant $\mathbb{1}[\cdot>0]$ from $\Phi$ is available in `scm/operators.py` but not wired as a preset.)*
   - **Regime-switching ablation**: a single deterministic T/2 break (`smoke_regime`, `RegimeSwitchNlinearSCMT`) and a genuine **HMM with $R\in\{2,3\}$ regimes** and random change-points (`smoke_regime_hmm`, `HMMRegimeSwitchNlinearSCMT`); used to evaluate Shift-VR under temporal non-stationarity (H7).

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

H3. *(Split 2026-07-14 — see `docs/hypotheses_assessment.md`; the original conflated a graph-error claim with a recourse-explainer claim, and a graph-discovery method emits no counterfactuals.)*
- **H3a (graph-error):** given a well-recovered causal graph (e.g. via DYNOTEARS, a self-graphing discovery baseline), the oracle structural CF derived from it retains near-full CF-faith — so standard explainers' CF-faith failures are *propagation* failures, not graph-estimation failures.
- **H3b (recourse, positive control):** a graph-aware *recourse* method (CARLA / PearlCARLA) achieves CF-faith $> 0.7$ with moderate validity/proximity degradation — reported as a positive control (faithful by construction), not independent evidence.

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
