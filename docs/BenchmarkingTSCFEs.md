# Benchmarking Counterfactual Explanations for Temporal Causal Data: Structural Challenges and Synthesis

The rapid integration of deep neural networks in safety-critical, sequential domains has highlighted the limitations of opaque decision-making models.[1, 2, 3, 4, 5] Whether in medical diagnostics, industrial process monitoring, or high-frequency financial forecasting, deep classifiers successfully map complex multidimensional inputs to accurate target classes, yet their internal decision paths remain hidden from human operators.[1, 4, 6, 7] The emerging discipline of Explainable Artificial Intelligence (XAI) seeks to resolve this tension by developing post-hoc interpretation frameworks.[6, 7, 8] Among these, counterfactual explanations are highly valued for their actionable nature.[1, 9, 10] Rather than simply highlighting correlations through static feature attribution or saliency maps, counterfactuals explore hypothetical "what-if" scenarios, identifying the minimal, highly localized modifications required in an input time series to alter a classifier's output to a desired target class.[1, 9, 10, 11]

Generating physically plausible and logically consistent counterfactual trajectories for multivariate, non-stationary time series remains exceptionally difficult.[1, 4, 12] Unlike static tabular data, temporal data is governed by complex temporal dependencies, dynamic feedback loops, and lead-lag causal constraints.[1, 13, 14, 15] When a post-hoc explainer perturbs a single time step or variable in isolation without propagating that change to its causal descendants over time, it violates the physical laws of the underlying system, producing mathematically valid but physically impossible trajectories.[1, 4, 16]

Evaluating whether temporal XAI methods respect these underlying causal laws is currently impossible.[17, 18] A fundamental benchmarking deficit exists: no public dataset or framework jointly provides ground-truth causal variables, a ground-truth lagged causal graph, and analytical ground-truth structural counterfactuals in a temporal classification setting.[17, 18] Without this unified ground truth, researchers cannot quantitatively measure the causal faithfulness of temporal counterfactual algorithms, leaving the field without a reliable mechanism to certify explainer safety and accuracy.[4, 19]

---

## Mathematical Formulations of Temporal Causal Dynamics

To rigorously define and evaluate whether a temporal counterfactual explanation respects physical dynamics, the generation process must be grounded in formal causal models.[10, 14, 20] Modern causal inference distinguishes between two primary mathematical frameworks: the Potential Outcome Model and the Structural Causal Model.[21] The Potential Outcome Model, widely utilized in statistical economics, frames causal inference as a missing data problem, relying on foundational assumptions such as the Stable Unit Treatment Value Assumption, positivity, and ignorability to estimate treatment effects by matching or imputing unobserved potential outcomes from observational distributions.[15, 21] However, this framework lacks graphical representations and struggles to model complex spat-temporal interactions or dynamic feedbacks among multiple variables.[21]

Conversely, the Structural Causal Model framework, formalized by Judea Pearl, provides a robust graphical language to represent data-generating mechanisms.[10, 21, 22] An SCM is defined as a tuple $M = \langle V, U, F, P(U) \rangle$, where $V$ is a set of endogenous (observed) variables, $U$ is a set of exogenous (noise) variables representing unobserved background factors, $F$ is a set of deterministic structural assignments $V_i := f_i(pa(V_i), U_i)$ where $pa(V_i) \subseteq V$ represents the causal parents of $V_i$, and $P(U)$ is a probability distribution over the exogenous noise factors.[14, 20, 22] The structural assignment symbol ":=" emphasizes the directional, causal nature of these equations, distinguishing them from simple algebraic equalities.[20, 23]

In dynamic physical systems, causal interactions unfold across time.[14, 20] A Temporal Structural Causal Model (TSCM) represents these relations by indexing variables by their temporal positions.[17, 20] For a multivariate time series $X_t = [X_t^{(1)}, X_t^{(2)}, \dots, X_t^{(M)}]$ at time step $t$, the structural assignment for each channel $j$ is written as:

$$X_t^{(j)} := f_j\!\left(pa\!\left(X_t^{(j)}\right), U_t^{(j)}\right) \quad \text{with} \quad U_t^{(j)} \sim P\!\left(U^{(j)}\right) \tag{1}$$

[14, 20]

The set of causal parents $pa\!\left(X_t^{(j)}\right)$ is restricted to lagged variables of the form $X_{t-\tau}^{(i)}$, where the time lag satisfies $0 \le \tau \le \tau_{\max}$.[20] Restricting $\tau \ge 0$ enforces the principle of temporal precedence—precluding causal influences from acting backward in time—while a finite maximum lag $\tau_{\max}$ bounds the memory of the dynamical system.[15, 20] The qualitative structure of these temporal parent-child relations across channels and lags is represented by a time-lagged causal Directed Acyclic Graph (DAG).[14, 20, 24]

To derive a true structural counterfactual trajectory under an SCM, one must execute the three-step causal ladder process [10, 21]:

### 1. Abduction

Given an observed, factual time-series sequence $X$, the latent, exogenous noise variables $U$ are inferred by inverting the structural equations $F$, such that $U_t^{(j)} = h^{-1}\!\left(X_t^{(j)}, pa\!\left(X_t^{(j)}\right)\right)$.[10, 21, 25] In continuous, multi-variable settings, normalizing flows (such as continuous normalizing flows over a causal DAG utilized in DoFlow) establish a bijective mapping between the observational space and the latent noise space, guaranteeing unique and exact recovery of the background state.[26]

### 2. Action

A specific intervention is introduced by replacing the structural equation of a target variable $X_{\mathcal{T}_{int}}^{(i)}$ at a designated intervention time $\mathcal{T}_{int}$ with a constant value $x'_{int}$, denoted mathematically using the do-operator as $do\!\left(X_{\mathcal{T}_{int}}^{(i)} = x'_{int}\right)$.[10, 25] This intervention breaks the natural causal pathways entering the target node while preserving all downstream structural relationships.[10, 14, 27]

### 3. Prediction

The dynamical system is forward-simulated by applying the updated structural assignments, utilizing the abduced exogenous noise variables $U$ and the intervened value $x'_{int}$.[10, 25] This generates the mathematically rigorous, causally consistent structural counterfactual sequence $X'_{CF}$.[25, 26] Because the downstream predictions are explicitly computed through the known structural equations, any perturbation applied to a parent variable will naturally and realistically propagate through all causal descendants across subsequent time steps.[10, 14, 25]

In continuous dynamical systems governed by non-linear differential equations, such as the Lotka-Volterra or Lorenz equations, evaluating the causal validity of these trajectories requires tracking Topological Causality (TC).[28] Topological Causality measures the directed effective influence between state variables, and any perturbation to a non-causal basis term in a system's governing equations will result in a sharp violation of this topological consistency.[28] Mathematically, evaluating TC across a pair of variables involves a nearest-neighbor search in the reconstructed phase space, exhibiting a computational complexity of:

$$O\!\left(T \cdot d \cdot \log T + T \cdot k \cdot d^3\right) \tag{2}$$

[28]

where $T$ represents the number of temporal embedding points, $k$ is the number of nearest neighbors, and $d$ is the embedding dimension.[28]

---

## The Benchmarking Deficit

Despite rapid advancements in temporal generative modeling, the XAI community has been severely constrained by a critical benchmarking deficit.[17, 18] No standardized benchmark suite jointly provides ground-truth causal variables, a ground-truth lagged causal graph, and analytical ground-truth structural counterfactuals in a temporal classification setting.[17, 18] Consequently, there is no quantitative way to measure whether a temporal explanation method respects causal structure or simply exploits model vulnerabilities and spurious, non-causal correlations to alter a label.[4] Existing resources only address isolated components of this multi-domain intersection, failing to establish a unified evaluation framework.[17, 18, 29]

For example, the widely utilized UCR and UEA time-series archives provide highly diverse classification tasks, but they lack any underlying causal equations, known graphs, or interventional data.[18, 29, 30] Conversely, physical benchmarks such as CauseMe, Lorenz-96 systems, and CausalRivers offer ground-truth lagged causal graphs for structure learning, but they strictly generate observational trajectories and completely lack downstream classification labels.[17, 18] Similarly, temporal generative frameworks like CausalTime and Temporal Causal-based Simulation (TCS) can learn functional dependencies and noise distributions from real observations to simulate realistic trajectories, yet they do not natively support analytical ground-truth counterfactuals under classification targets.[24, 31, 32] This structural disconnect forces researchers to rely on non-causal surrogate metrics—such as Euclidean proximity or reconstruction loss—which actively penalize true causal counterfactuals, as these physical trajectories require widespread, non-sparse downstream propagation to remain structurally consistent.[1, 4, 16, 25]

---

## Taxonomy of Temporal Counterfactual Explainers

Current algorithms for generating counterfactual explanations on sequential data employ diverse optimization, heuristic search, and generative architectures.[1, 11, 33] These methods can be systematically categorized into four primary paradigms, each exhibiting distinct structural limitations when evaluated against temporal causal consistency.[1, 11, 12]

### Instance Substitution and Nearest Neighbor Techniques

These frameworks, such as Native Guide and CoMTE, construct counterfactuals by identifying a "nearest unlike neighbor" from the training dataset that belongs to the target class.[11] They then substitute subsequences of the original query instance with corresponding segments from the donor instance, often guided by differentiable saliency maps or classifier feature-weight vectors.[11, 12] MASCOTS extends this paradigm by converting multivariate series into a symbolic Bag-of-Receptive-Fields (BoRF) representation to identify and modify key symbolic sub-patterns.[9]

While these methods guarantee statistical plausibility by using real data segments, the direct splicing of subsequences creates severe physical discontinuities.[1, 11] The inserted segment reflects the causal dynamics of the donor system under its own unique background conditions, which are structurally incompatible with the latent exogenous noise $U$ of the recipient system.[1, 21] This mismatch violates the SCM consistency assumption, resulting in hybrid trajectories that represent physically impossible states.[21]

### Evolutionary and Heuristic Search Algorithms

Genetic and evolutionary search algorithms, including TSEvo, Sub-SpaCE, and CONFETTI, treat counterfactual generation as a multi-objective optimization problem.[11, 34, 35] They initialize a population of candidate sequences and iteratively apply customized crossover, mutation, and selection operators in the input space or within a low-dimensional autoencoder latent space.[11, 34, 35] The fitness functions balance prediction confidence, sparsity, and reconstruction error to guide the population toward the target class.[11, 34, 35]

However, because these search heuristics operate strictly on statistical distance losses and lack an explicit temporal transition model, they mutate parent and child variables independently.[1, 4, 11] By failing to propagate perturbations down the known physical channels, they generate highly fragmented sequences that break basic lead-lag relationships.[1, 4]

### Deep Latent Space and Generative Models

Generative approaches utilize deep neural networks, such as Autoencoders, TimeGANs, continuous normalizing flows, or Variational Autoencoders, to project time series into a continuous latent space.[11, 25, 26] Models like LatentCF++, Glacier, and TNCM-VAE optimize the latent representation to flip the classifier's prediction while enforcing plausibility through reconstruction constraints.[11, 25, 36] DoFlow maps these dynamics directly over a causal DAG using continuous normalizing flows.[26]

While these frameworks successfully capture global temporal patterns, they are fundamentally optimized on observational data distributions.[1, 10, 26] An intervention $do(X_t^{(j)} = x'_{int})$ represents an out-of-distribution event that breaks the natural statistical correlations observed in factual data.[10, 27] Standard generative models struggle to accurately simulate these out-of-distribution interventions, frequently reverting to spurious, associative correlations that fail to reflect the true, physical causal mechanisms of the system.[4, 10]

### Amortized and Controllable Intervention Architectures

To bypass the high computational latency and instability of instance-wise optimization, modern architectures learn global, shared intervention strategies.[12, 37, 38] ConTex parameterizes interventions through two dedicated heads: a temporal relevance mask that identifies where to modify, and a modification strength head that determines how to modify, generating counterfactuals in a single forward pass.[12] CLEF utilizes a sequence encoder and a condition adapter to execute highly localized sequence editing at specific future time points, preserving unaffected variables.[38] Similarly, CELT learns a structured latent space where target-class instances form tight clusters, guiding segment-wise, time-local edits.[37]

Although these frameworks provide real-time, highly controllable editing capabilities, they are trained on standard classification losses without explicit SCM regularizers.[4, 12, 38] Consequently, their learned masks and adapters cannot be validated for causal faithfulness, leaving them vulnerable to severing real physical dependencies.[4]

Furthermore, post-hoc explanation tools introduce severe dual-use concerns and cybersecurity risks.[39] While they assist operators in debugging models, a motivated adversary can exploit the highly detailed feature-importance rankings and counterfactual pathways provided by explainers to reconstruct the classifier's decision boundaries.[39] This information can facilitate highly powerful membership inference, model extraction, and localized adversarial evasion attacks (such as poisoning or backdoors) by targeting the model's most sensitive causal pathways.[39]

The comparative mechanics and failure modes of these explainer paradigms are systematically structured in **Table 1**.

---

## Empirical Performance and Comparative Benchmarking

Integrating causal priors into sequence modeling significantly enhances both classification accuracy and explanation reliability.[4, 30] To quantify these effects, several benchmarking evaluations have been conducted across standard datasets, including the University of East Anglia (UEA) multivariate time series, the UCR archive, MIMIC-IV, PhysioNet 2019 Sepsis, and Kinetics-400.[4, 30, 36]

An extensive evaluation of pretraining paradigms was executed using the TabPFN v2.5 backbone across 75 UCR and UEA datasets.[30] The datasets were restricted to those fitting within TabPFN's native operating parameters (number of classes $K \le 10$ and flattened column product $M \cdot T \le 500$).[30] The evaluation compared standard TabPFN against a custom preprocessing pipeline (incorporating patch-aligned tokenisation and channel-specific z-scoring) and models finetuned on CAUKER-generated synthetic causal priors.[30, 40] During prior-data synthesis, the training sequences were subjected to realistic post-processing augmentations, including Kumaraswamy value-level warps to introduce non-linear distortions, strided pooling to alter temporal granularity, predictive truncation to simulate incomplete observations, and random value-level missingness.[30] The empirical results, summarized across two ensemble sizes ($e \in \{1, 8\}$), are detailed in **Table 2**.

Finetuning the transformer backbone on the full, causally structured CAUKER prior—which simulates random DAGs and non-linear interactions across variables—statistically outperforms all other configurations, establishing a new state-of-the-art rank on the critical-difference diagrams over the 75 benchmarks.[30, 40] Conversely, finetuning on the tabular-only ablation (D) degrades performance, demonstrating that exposing sequence models to causally coherent temporal structures is essential for robust out-of-distribution generalization.[30]

Beyond classification accuracy, establishing causal and mechanistic consistency within temporal explainers dramatically reduces fragility under distribution shifts, sensor deterioration, and behavioral drift.[4] Standard post-hoc explainers frequently produce contradictory attributions when evaluated on non-stationary streams.[4] To address this, a unified framework was developed that integrates causal interventions, shift-invariant regularization at the explanation level, and mechanistic confirmation via Internal Ablation.[4]

When evaluated on the MIMIC-IV and PhysioNet 2019 Sepsis datasets, this causally stable framework achieved a 19% relative increase in Rank Correlation ($\text{RankCorr} = 0.82$) and improved out-of-distribution AUROC from 0.753 to 0.789 under temporal distribution shifts, proving that predictive robustness alone cannot guarantee the reliability of explanations.[4]

Further empirical evaluations highlight critical performance trade-offs in clinical and industrial monitoring tools.[2, 36, 41] In action recognition tasks using video transformer models (which integrate spatial and temporal features), the Spatio-Temporal Attention Attribution (STAA) model simultaneously extracts frame-level and sequence-level relevance from attention weights in a single forward pass, achieving a faithfulness score of $0.844 \pm 0.116$ and a monotonicity score of $0.850 \pm 0.030$ on the Kinetics-400 dataset while operating within sub-100ms real-time latency parameters.[36]

In anomaly detection domains, the Explainable Anomaly Classification Tool (EXACT) benchmarks Extreme Gradient Boosting (XGBoost), Decision Trees, and LSTM Autoencoders (LSTM-AE) alongside SHAP, LIME, and DiCE.[2] XGBoost offers the most balanced predictive performance, while LSTM-AE is computationally intensive but successfully captures complex temporal patterns.[2] When evaluating explanations via Normalised Discounted Cumulative Gain (NDCG), SHAP provides highly stable and robust feature rankings but suffers from severe computational latency, whereas LIME is computationally faster but yields highly unstable attributions.[2]

These multi-dimensional metrics and verification protocols are compiled in **Table 3**.

---

## Blueprint for a Joint Temporal Causal Classification Benchmark

To resolve the benchmarking deficit, a standardized synthetic framework must be developed.[17, 18] This benchmark must simultaneously generate a Temporal Structural Causal Model, a categorical classification label governed by a latent causal state, and the corresponding analytical ground-truth structural counterfactuals.[18, 25, 42] The construction of this benchmark is executed in four sequential phases:

### Phase 1: Synthetic TSCM Prior Generation

The framework first samples a dynamic, time-lagged causal DAG $G$ over $M$ channels up to a maximum lag $\tau_{\max}$.[20, 24] The structural causal mechanisms are analytically sampled from a family of non-linear autoregressive functions using a dictionary of mathematical operators:

$$X_t^{(j)} := \sum_{X_{t-\tau}^{(i)} \in pa\!\left(X_t^{(j)}\right)} w_{ij} \cdot \phi_{ij}\!\left(X_{t-\tau}^{(i)}\right) + U_t^{(j)} \tag{3}$$

[18, 20]

where the operator $\phi_{ij}$ is selected uniformly from:

$$\{\text{identity},\ \sin,\ \cos,\ \tanh,\ |\cdot|,\ (\cdot)^2,\ \exp(-|\cdot|)\}$$

and the coefficients are drawn from Gaussian distributions $w_{ij} \sim \mathcal{N}(0, \sigma^2_w)$.[18] The exogenous noise terms $U_t^{(j)}$ are sampled from independent, non-Gaussian distributions to ensure the identifiability of the causal structure.[14, 22] To introduce realistic non-stationarity, the prior supports regime-switching dynamics driven by a hidden Markov process, which triggers structural breaks and modifies the active coefficient weights $w_{ij}$ over time.[16, 17, 18]

### Phase 2: Label Generation via Latent Mechanisms

Unlike standard forecasting models, the prior must generate a categorical target label $Y \in \{1, \dots, K\}$ for each multivariate sequence $X$.[30, 42] This is governed by the principle of **Label Visibility**:

> The target class $Y$ is structurally determined by a latent state transition or threshold crossing of a key causal variable within the SCM, ensuring that the label arises directly from a meaningful causal mechanism.[42]
>
> Although the label is generated by hidden latent states, its physical effects must propagate downstream, remaining fully recoverable and inferable from the observed, multi-channel trajectories.[42]

Pretraining experiments demonstrate the validity of this approach.[42, 43] A model pretrained on a hidden-label SCM prior with an enforced label-observation path (such as TIC-FM) achieved an accuracy of 0.776 under full-support inference and 0.802 accuracy under retrieval-64 settings on the 42-task binary UCR subset, proving that sequence classifiers can successfully learn these latent mappings.[42, 43]

### Phase 3: Analytical Counterfactual Validation

For any generated time-series instance $X$ and a user-specified intervention $do\!\left(X_{\mathcal{T}_{int}}^{(i)} = x'_{int}\right)$, the benchmark analytically derives the unique ground-truth counterfactual trajectory $X'_{CF}$.[25, 26] Because the additive structural equations are exactly invertible, the latent exogenous noise is abduced as:

$$U_t^{(j)} = X_t^{(j)} - \sum_{X_{t-\tau}^{(i)} \in pa\!\left(X_t^{(j)}\right)} w_{ij} \cdot \phi_{ij}\!\left(X_{t-\tau}^{(i)}\right) \tag{4}$$

[14, 18, 20]

The intervention $do\!\left(X_{\mathcal{T}_{int}}^{(i)} = x'_{int}\right)$ is then applied, and the system is forward-simulated utilizing the abduced noise sequence $U$ and the structural equations.[10, 25, 26] This generates a singular, mathematically precise counterfactual trajectory $X'_{CF}$ that serves as the absolute target for evaluation.[25, 26]

### Phase 4: Unified Evaluation Suite

The generated dataset is passed to a target black-box classifier $f(X)$.[11, 44] When a post-hoc explainer generates a counterfactual sequence $X'_{exp}$ to alter the model's prediction to $y^*$, its performance is evaluated across a multi-dimensional metric suite [4, 11, 25]:

> **Superseded (2026-07-15).** This section records the *original* design. The
> implemented protocol is defined in `docs/updated_general_plan.md` §Axis C and
> reconciled in `docs/spec_code_reconciliation.md` §3–§4. Two changes matter
> here: the DTW/MAE-to-oracle faithfulness measure below was **discarded** in
> favour of mechanism-residual CF-faith under two semantics, and TRSI's role was
> narrowed from a coherence *criterion* to a mechanism-free *descriptor*.

- **Causal Faithfulness:** *(discarded — see the note above.)* Originally measured as the Mean Absolute Error or Dynamic Time Warping distance between $X'_{exp}$ and the analytical SCM counterfactual $X'_{CF}$.[25, 26, 38] This was intended to quantify whether the explainer propagates the physical consequences of the intervention downstream.[4, 25] It was replaced because distance-to-a-single-oracle commits to one privileged trajectory and cannot express the rollout-vs-Pearl distinction; the implemented CF-faith tests consistency with the *mechanism* instead.
- **Temporal Coherence:** *(reframed — see the note above.)* Evaluated via the Temporal Relevance Smoothness Index (TRSI) to check that generated changes do not introduce unfeasible, step-to-step discontinuities.[1, 3] TRSI is adopted from prior work and is mechanism-free: it is a complementary descriptor of edit smoothness, **not** evidence of causal faithfulness (that is CF-faith's job).
- **Explanation Stability:** Measured using Rank Correlation under temporal distribution shifts to verify the robustness of the explainer's decision path.[4]

---

## Strategic Directions for Causal Temporal Explainability

Grounding temporal explainability in rigorous causal models is essential for developing safe and trustworthy AI systems.[4, 19] To advance the field toward true causal trustworthiness, research should prioritize three key strategic areas:

### Integration of Causal Discovery and Post-Hoc Explanations

Explainability frameworks must transition away from purely correlational models.[4, 45] In environments where the physical causal graph is not known a priori, algorithms should incorporate an upstream temporal causal discovery step—using constraint-based methods like PCMCI or score-based frameworks like Dynotears—to infer a time-lagged causal prior from observational data.[14, 16] This learned graph must be integrated directly into the counterfactual optimization objective as a structural constraint.[14, 28] By penalizing perturbations that deviate from the inferred topological causality, the resulting counterfactual trajectories can be regularized to remain consistent with the physical dynamics of the system.[28]

### Transitioning to Amortized, Causal-Prior Training

The high computational cost and optimization instability of instance-wise counterfactual generation make it impractical for real-time applications, such as clinical ICU monitoring or autonomous driving.[12, 36] Future research should focus on training amortized, global intervention models (such as ConTex or CLEF) directly on synthetic causal priors.[12, 18, 38] Pretraining transformer-based architectures on diverse, multi-modal structural causal priors (similar to the PFN pretraining paradigm) enables these models to perform in-context causal effect estimation and generate causally consistent counterfactuals in a single forward pass at test time, bypassing the need for slow, iterative instance-level optimization.[12, 18, 30, 43]

### Implementing Rigorous Certification and Mechanistic Verification

To deploy temporal deep learning models in safety-critical settings, their post-hoc explanations must undergo formal, multi-step certification.[4, 19] Explainers must be subjected to rigorous mechanistic verification protocols—such as activation patching and causal tracing—to confirm that the temporal features and channels highlighted by the explainer align with the actual internal computational pathways utilized by the model to produce its predictions.[4, 46] By establishing and jointly optimizing causal validity, out-of-distribution stability, and internal mechanistic consistency, the AI community can transition from fragile, correlational explanations to causally anchored, robust, and verifiable temporal XAI.[4]

---

## References

> **Verification note (2026-07-08):** all 46 references below were checked against live sources by
> Literature Intelligence and reviewed by the PI; corrections (title/URL fixes, aggregator/duplicate/
> unverifiable flags) are annotated inline per entry. Two confirmed duplicate pairs (#1/#33, #17/#18)
> are flagged rather than collapsed, to avoid renumbering every inline `[N]` citation elsewhere in this
> document by hand — collapse them with proper citation-management tooling during manuscript
> preparation (M7), not by manual find-replace. Full findings, evidence, and the novelty-ledger analysis:
> `docs/references_verified.md`.

1. [What-If Explanations Over Time: Counterfactual Explanations for Time Series Classification](https://arxiv.org/html/2603.27792v1) — arXiv (Schlegel & Seidl, 2026; title corrected 2026-07-08 from "Counterfactuals" to "Counterfactual Explanations." **Duplicate of ref #33** — same arXiv ID, two URL formats; kept as separate numbers to avoid renumbering every inline citation, flagged for collapse during manuscript preparation — see `docs/references_verified.md`.)
2. [EXPLAINABLE AI FOR TIME SERIES ANOMALY DETECTION](https://www.diva-portal.org/smash/get/diva2:1972404/FULLTEXT01.pdf) — Diva-Portal.org
3. [Regularizing Temporal Explanations in Dynamic Neural Networks](https://www.mdpi.com/2079-9292/15/10/2200) — MDPI (**Unverifiable as of 2026-07-08**: direct fetch and the issue's table of contents both returned HTTP 403; no independent corroboration found for this specific article number. Recommend confirming directly in-browser before relying on this citation — see `docs/references_verified.md`.)
4. [Explanation-level Robustness for Temporal Deep Learning Under Distribution](https://inass.org/wp-content/uploads/2026/03/2026063026-2.pdf) (Verified 2026-07-08, author-confirmed live at this URL on the journal's own domain after an initial automated-fetch attempt returned domain-wide 502s — see `docs/references_verified.md`.)
5. [Estimating Feature Attributions for Time Series Classification: Supervised Prediction of Integrated Gradients under the Multitas](https://www.diva-portal.org/smash/get/diva2:1876681/FULLTEXT01.pdf) — Diva-Portal.org
6. [A Survey of Explainable Artificial Intelligence (XAI) in Financial Time Series Forecasting](https://arxiv.org/abs/2407.15909) — arXiv / ACM Computing Surveys, DOI 10.1145/3729531 (Arsenault, Wang, Patenaude. Corrected 2026-07-08 to the canonical location — previously linked a third-author company-site PDF mirror whose upload path could mislead a reader into thinking this is 2026 work; it is 2024/2025 work — see `docs/references_verified.md`.)
7. [The application of explainable artificial intelligence in the prediction, diagnoses, treatment, and management of chronic diseases: A systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12647564/) — PMC
8. [C-SHAP for time series: An approach to high-level temporal explanations](https://arxiv.org/html/2504.11159v1) — arXiv
9. [MASCOTS: Model-Agnostic Symbolic COunterfactual explanations for Time Series](https://arxiv.org/html/2503.22389v1) — arXiv
10. [Causal and Counterfactual Explanations](https://www.emergentmind.com/topics/causal-and-counterfactual-explanations) — Emergent Mind (**Flagged 2026-07-08**: AI-generated topic-aggregation page, not a primary scientific source — synthesizes roughly 17 arXiv papers. Recommend citing the primary papers it draws on directly instead before final submission — see `docs/references_verified.md`.)
11. [Towards plausibility in time series counterfactual explanations](https://arxiv.org/html/2603.08349v1) — arXiv
12. [ConTex: Reformulating Counterfactual Generation For Time Series Forecasting](https://arxiv.org/html/2606.18049) — arXiv
13. [Counterfactual Explanations for Time Series Should be Human-Centered and Temporally Coherent in Interventions](https://arxiv.org/html/2512.14559v1) — arXiv
14. [Causally-Constrained Probabilistic Forecasting for Time-Series Anomaly Detection](https://arxiv.org/html/2604.17998v1) — arXiv
15. [Time Series Causal Inference](https://donskerclass.github.io/CausalEconometrics/TimeSeries.html) — Rachel Leah Childers
16. [From Causal Discovery to Dynamic Causal Inference in Neural Time Series](https://arxiv.org/html/2603.20980v1) — arXiv
17. [Interventional Time Series Priors for Causal Foundation Models](https://arxiv.org/html/2603.11090v1) — arXiv (cited a second time as ref #18 via a different URL format — see `docs/references_verified.md`.)
18. [Interventional Time Series Priors for Causal Foundation Models (PDF)](https://arxiv.org/pdf/2603.11090) — arXiv (**Duplicate of ref #17** — same paper, PDF-format URL instead of the abs-page URL, confirmed 2026-07-08. Kept as a separate number to avoid renumbering every inline citation, flagged for collapse during manuscript preparation — see `docs/references_verified.md`.)
19. [Explainable AI: Learning from the Learners](https://arxiv.org/html/2601.05525v2) — arXiv
20. [Causal inference for time series](https://doi.org/10.1038/s43017-023-00431-y) — Nature Reviews Earth & Environment, 2023, vol. 4, pp. 487–505 (Runge, Gerhardus, Varando, Eyring, Camps-Valls; a publisher correction was later issued at DOI 10.1038/s43017-023-00471-4. Corrected 2026-07-08 to the canonical Nature DOI — previously linked a personal-site mirror of the accepted manuscript — see `docs/references_verified.md`.)
21. [Spatio-temporal graphical counterfactuals: an overview](http://scis.scichina.com/en/2026/141201.pdf)
22. [Structural Causal Model Essentials](https://www.emergentmind.com/topics/structure-causal-model) — Emergent Mind (**Flagged 2026-07-08**: AI-generated topic-aggregation page, not a primary source — synthesizes roughly 11 arXiv papers. Recommend citing Pearl directly (e.g. Pearl, *Causality*, 2009) for SCM foundations instead — see `docs/references_verified.md`.)
23. [An Extended Class of Instrumental Variables for the Estimation of Causal Effects](https://economics.ucr.edu/wp-content/uploads/2019/11/HalWhite_Sem_6-2-06.pdf) — UCR Department of Economics
24. [CausalTime: Realistic Time-Series Generation for Benchmarking Causal Discovery](https://arxiv.org/abs/2310.01753) — arXiv (Cheng, Wang et al., ICLR 2024; corrected 2026-07-08, was a broken Hugging Face search-query link — see `docs/references_verified.md`)
25. [Towards Causal Market Simulators](https://arxiv.org/html/2511.04469v4) — arXiv
26. [DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting on Time Series](https://arxiv.org/abs/2511.02137) — arXiv (also OpenReview id=4IPIhOgVqz. Full title restored 2026-07-08 — the draft previously dropped "Forecasting on Time Series." Acceptance/venue decision not independently confirmed — see `docs/references_verified.md`.)
27. [Methods in causal inference. Part 1: causal diagrams and confounding](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588567/) — PMC / NIH
28. [CaNDiCE: Causal Discovery of Nonlinear Dynamics Through Constraints](https://openreview.net/forum?id=bgdTK6cniJ) — OpenReview (**Exact title unconfirmed as of 2026-07-08** — one source suggests "...Through Constraints" as cited here, another suggests "...through Counterfactual Explanations." Recommend confirming directly in-browser before final citation — see `docs/references_verified.md`.)
29. [ExplainTS: a benchmark dataset of pretrained models and post-hoc explanations for time-series classification](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1759110/full) — Frontiers
30. [A Causal DAG Prior for Synthetic Time-Series Classification Datasets](https://arxiv.org/html/2606.21776v1) — arXiv
31. [Temporal Causal-based Simulation for Realistic Time-series Generation](https://arxiv.org/html/2506.02084v1) — arXiv (Gkorgkolis, Kougioulis, Wang, Caglayan, Tonon, Simionato, Tsamardinos, 2025; v1 title, matching this citation. **Note, 2026-07-08**: the live v2 (12 May 2026) has been retitled "Adversarial Causal Tuning for Realistic Time-series Generation" — same authors/method, TCS→ACT rebrand; a reader following this link today sees a different title than quoted here — see `docs/references_verified.md`.)
32. [GitHub — gkorgkolis/TCS: Temporal Causal-based Simulation (TCS)](https://github.com/gkorgkolis/TCS)
33. [What-If Explanations Over Time: Counterfactuals for Time Series Classification](https://arxiv.org/abs/2603.27792) — arXiv (**Duplicate of ref #1** — identical arXiv ID 2603.27792, different URL format, confirmed 2026-07-08. Kept as a separate number to avoid renumbering every inline citation, flagged for collapse during manuscript preparation — see `docs/references_verified.md`.)
34. [CounterBench: Evaluating and Improving Counterfactual Reasoning in Large Language Models](https://arxiv.org/html/2502.11008v1) (Chen, Singh, Ma, Tang. Title corrected 2026-07-08 — was cited as "A Benchmark for Counterfactuals Reasoning in Large Language Models." Note: this paper is about LLM counterfactual reasoning over formal causal graphs, not time-series counterfactual explanation — relevance to this draft's argument should be double-checked when the manuscript is written — see `docs/references_verified.md`.)
35. [TSEvo: Evolutionary Counterfactual Explanations for Time Series Classification](https://www.researchgate.net/publication/369483885_TSEvo_Evolutionary_Counterfactual_Explanations_for_Time_Series_Classification) — ResearchGate
36. [STAA: Spatio-Temporal Attention Attribution for Real-Time Interpreting Transformer-based AI Video Models](https://www.researchgate.net/publication/392352868_STAA_Spatio-Temporal_Attention_Attribution_for_Real-Time_Interpreting_Transformer-based_AI_Video_Models) — ResearchGate (Wang, Liu; *IEEE Access*, vol. 13, 2025, pp. 101647–101661; also arXiv 2411.00630. Authors added 2026-07-08 — previously omitted — see `docs/references_verified.md`.)
37. [Counterfactual Explanations via Latent Structure for Time Series Classification](https://openreview.net/forum?id=oKUjFiuoVi) — OpenReview (**Flagged 2026-07-08**: existence corroborated indirectly; acceptance/publication status could not be confirmed — treat as an anonymous/active submission, not a published result, until confirmed otherwise — see `docs/references_verified.md`.)
38. [Controllable Sequence Editing for Counterfactual Generation](https://arxiv.org/abs/2502.03569) — arXiv (Li, Li, Ektefaie, Messica, Zitnik; Harvard/Zitnik Lab; code at github.com/mims-harvard/CLEF. Corrected 2026-07-08 to the canonical arXiv ID — previously cited only the lab project page. Status "In Review" as of 2025, not a published/accepted paper — see `docs/references_verified.md`.)
39. [Temporal-XAI: An Attention-Based Explainable Framework for DoS Attack Detection with Temporal Pattern Analysis](https://www.researchgate.net/publication/396533379_Temporal-XAI_An_Attention-Based_Explainable_Framework_for_DoS_Attack_Detection_with_Temporal_Pattern_Analysis) — ResearchGate
40. [CAUKER: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data](https://helios2.mi.parisdescartes.fr/~themisp/publications/iclr26-cauker.pdf) — Paris Descartes
41. [Navigating Time's Possibilities: Plausible Counterfactual Explanations for Multivariate Time-Series Forecast through Genetic Algorithms](https://arxiv.org/html/2603.00855v1) — arXiv (Zuin, Veloso. Full title restored 2026-07-08 — the draft previously dropped the "Navigating Time's Possibilities:" prefix. Note: forecasting-target method, same caveat as ref #12 — see `docs/references_verified.md`.)
42. [Synthetic Causal Priors for In-Context Time-Series Classification](https://openreview.net/forum?id=nMRrgMF2S8) — OpenReview (**Unverifiable as of 2026-07-08**: direct fetch blocked; extensive search returned no independent corroboration of this title or ID anywhere. Recommend confirming directly in-browser before citing — see `docs/references_verified.md`.)
43. [Beyond Task-Specific Classifiers: In-Context Inference for Time Series Classification Foundation Models](https://openreview.net/forum?id=HVvARHEA9M) — OpenReview (**Flagged 2026-07-08, Medium confidence**: appears to be the same work as arXiv 2602.00620, "Rethinking Zero-Shot Time Series Classification: From Task-specific Classifiers to In-Context Inference" [Fang, Xie, Nie, Ling, Liu, Li, Zhang, Pan, Palpanas, Cai] — plausibly a workshop-title variant, a common and benign pattern, but the OpenReview ID could not be directly confirmed to resolve to this exact paper — see `docs/references_verified.md`.)
44. [TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series Models](https://arxiv.org/html/2601.09776v1)
45. [Explaining Time Series Classification Predictions via Causal Attributions](https://www.computer.org/csdl/proceedings-article/ictai/2025/491900a042/2ct0OHj6XZK)
46. [Mechanistic Interpretability for Transformer-based Time Series Classification](https://arxiv.org/html/2511.21514) — arXiv
