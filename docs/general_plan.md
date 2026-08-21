# CausalTemp-XAI — General Plan

**Status:** single source of truth for the scientific claim, scope, and
evaluation protocol. Supersedes the deleted `01_vision.md`,
`updated_general_plan.md`, `mvp_plan.md`, `PROJECT_PLAN.md`.
**Last substantive revision:** 2026-08-12 (M4e re-derived, H2 refuted, H2′
added).

**Method naming revision (2026-08-21):** new APIs and result schemas use
`NoiselessSCMRecourse` and `PearlSCMRecourse`. Run artifacts produced before this change retain
the historical labels `CARLA` and `PearlCARLA`; those artifacts are evidence, not migration
inputs, and are not rewritten.

This document states *what the claim is* and *how it is measured*, and is
written to stand alone. Pre-registration detail, thresholds, milestone status
and venue strategy live in planning documents kept outside the repository;
nothing here depends on them.

---

# 1. Motivation

CF explanation methods answer the question practitioners actually ask in
high-stakes sequential settings: *what would have had to be different in this
trajectory for the model to predict otherwise?* The temporal CF literature
scores its answers with validity, proximity, sparsity and OOD-plausibility —
every one of which measures the CF against **the model**, none against **the
process that generated the data**.

That gap is unmeasurable without a benchmark that knows its own mechanism.
This work builds one and uses it to ask: *when a CF method claims a change
would produce an outcome, does the world agree?*

On every wired method the answer is **no** or **only vacuously yes** —
separating those two is the instrument's job. One method's claim is denied
outright (`Δ_total = 0.39`). The rest are "agreed with" only because they
declared 83–100 of a 100-step trajectory to be interventions, leaving the
mechanism nothing to predict. A single-`do()` recourse control is the only
method proposing an intervention in the usual sense of the word.

# 2. Scope

> **A benchmark and audit protocol for counterfactual explanation methods on
> time series, evaluated on whether their claimed causal efficacy is realised
> by the data-generating process.**

Narrowed 2026-07-29, re-prioritised 2026-07-31:

- **CF explainers only.** Attribution (TimeSHAP, Dynamask) and concept-based
  methods (TCAV-T, CBM-T, LEAP, iVAE, β-VAE) are out of scope. The concept
  axis was deleted 2026-08-03 with its metric module and method code; axis
  letters were reassigned (§5). **Concept stability** is dropped with it.
- **Causal-graph access is an assumption, not a discovery problem.** On the
  synthetic tiers the graph is the generator's ground truth. Discovery is
  load-bearing only in tier 3, where one graph is propagated to both methods
  and metrics — never a different graph for generation and evaluation.
- **Robustness is its own axis**, not a grab-bag shared with attribution
  diagnostics.

Proposing a method that scores well against this protocol is the natural
follow-up. Nothing here depends on it.

# 3. Gap in the Literature

- **Temporal XAI benchmarks** (CFTS, XTSC-Bench, AMEE, CARLA): validity,
  proximity, sparsity, OOD — no causal faithfulness, no way to ask whether a
  proposed intervention *works*.
- **Temporal causal discovery benchmarks** (CausalDynamics, CausalTime, CITRIS):
  graph recovery, no explanation method in the loop.
- **Necessity/sufficiency for explanation** (LEWIS; Watson et al. 2021): the
  right question, asked in the static setting, necessarily answered with
  Tian–Pearl *bounds* since both potential outcomes are never jointly observed.
- **SCM-regularised temporal CF** (Bahri et al., IEEE BigData 2025): the SOTA
  in causally-motivated temporal CF. It uses the SCM to **penalise** implausible
  edits — a residual `‖T'_t − f(Pa(T'_t))‖` added to a Wachter-style objective
  over a free perturbation `T' = T + Δ`. No `do()`, no intervention time, no
  abduction.

**Related Work framing:** *the field uses SCMs to **penalise** counterfactuals;
nobody uses them to **derive** the counterfactual a proposal is scored against.*

**The delta:** in the temporal additive-noise setting abduction is exact
subtraction, so both potential outcomes are computable and necessity/sufficiency
is **exact rather than bounded** — plus a model-vs-world decomposition
separating "the CF is not what the world would produce" from "the classifier
does not track the true label rule".

> **Penalise-vs-derive is load-bearing, not semantic.** A residual penalty is a
> *local one-step* check: it asks whether `T'_t` is close to `f` of `T'`'s own
> parents at each `t` independently. It never propagates the intervention and
> never abducts the factual noise. So a CF that is locally mechanism-consistent
> at every timestep while its intervention has **no effect on the outcome**
> scores perfectly. That is the NoiselessSCMRecourse pathology measured here on 2026-07-30
> (CF-faith 1.00, validity 1.00, intervention effect 2.6e-04): the strongest
> published causal-CF criterion is structurally blind to the failure mode §4.1
> measures.

**Answering Bahri et al. §V direction 1 — partial causal knowledge (M4e).**
The benchmark grades partial causal knowledge against known truth: degrade the
true graph by a controlled fraction, measure how much CF-faith the oracle CF
loses when derived from the wrong graph (`graph_error`) versus from failing to
propagate (`propagation_error`, zero for the oracle).

> **M4e's dissipation-dependent finding was withdrawn 2026-08-12.** It rested on
> two defects, each sufficient alone: the pre-P0-2 mechanism left the true graph
> carrying **0.37%** of `full_nl`'s observed variance (so its flat graph-quality
> ladder measured the benchmark's own degeneracy, not dissipation), and the
> cross-family ratio compared differences of `exp(−residual)` soft scores across
> families whose state σ differs ~6×.
>
> Re-derived post-retune (graph share **91.13%**), in units of each config's own
> state σ, 3 seeds, at full graph corruption:
>
> | config | graph-axis span | method-axis span | method/graph |
> |---|---|---|---|
> | `full_nl` (ρ = −0.689) | 1.119 σ | 2.171 σ | 1.9× |
> | `smoke_spring` (ρ = +0.024) | 0.082 σ | 0.982 σ | 12.0× |
>
> The ordering **inverts**: the dissipative family is ~13.7× *more*
> graph-sensitive, and the retracted "45×–123×" figure has no successor in the
> same direction. The family labels are not the error — the retune made
> `full_nl` *more* dissipative (ρ from −0.372 to −0.689), and the hypothesis
> predicts a flatter ladder for a more strongly contracting family, whereas the
> ladder moved ~226× the other way. A degenerate mechanism, not dissipation,
> explains the original flat result. `graph_error` is a difference of soft
> scores and must never be compared across configs without its
> `graph_error_sigma` companion.

**Independent confirmation that the gap is open (2026-07-31).**
CausalTimePrior (Thumm & Chen, arXiv 2603.11090) builds a prior over temporal
SCMs and surveys the generator landscape. Its Appendix B draws the distinction
this benchmark turns on, in the authors' words: their observational and
interventional runs *"draw independent noise realizations"*, answering the
**interventional** query `P(X | do(·))`, not the **counterfactual**
`P(X^CF | X^obs, do(·))`. Generating counterfactual pairs would require sharing
the noise tensor — which they call a *"natural extension"* and *"a strictly
harder task than interventional prediction"*. That is what
`benchmarks/structural_cf.py` already does.

> **Novelty caution (RISK-06).** The *infrastructure* claim is partly contested:
> CAUKER (ICLR 2026) and arXiv 2606.21776 provide ground-truth-DAG +
> classification labels, and DoFlow provides analytical structural CFs for
> forecasting. The surviving delta is structural ground truth **as an evaluation
> oracle for independently-produced, post-hoc CF explainers** — which is why §4
> leads with the instrument and the finding, not the artifact.

# 4. Contributions

Ordered by strength. Each is defensible if the others are challenged, and
**none requires any evaluated method to succeed.**

**1. A model-vs-world audit instrument for temporal CF explanations.** For a
proposed CF, the same intervention is realised by the true mechanism and the
two outcomes compared, decomposing the disagreement additively into a
trajectory term and an outcome term. Necessity and sufficiency are computed
**exactly**, not bounded.

Construct validity is part of the contribution, because the audit's answer
depends on **how the proposal is read** and neither endpoint is trustworthy
alone: reading a CF as a single `do()` at `t0` inflates the trajectory term for
densely-editing methods; reading every edited slice as an intervention drives it
to zero *by construction*. So the benchmark reports the bracket — **do-complexity
`D`**, the number of timesteps a proposal must declare as actions before the
mechanism can reproduce it, published beside `Δ_trajectory` (RISK-18).

> **Headline finding, restated 2026-08-03.** The earlier headline
> ("no wired method makes a causal intervention, `Δ_trajectory ≈ 1.0`") was
> substantially an artifact of the single-slice reading and is **retracted**.
> Under the whole-proposal reading `Δ_trajectory` is 0.00 for every method and
> `Δ_total` falls from 0.91–0.99 to 0.02–0.39. What replaces it is not weaker:
> **methods buy their claimed validity by intervening almost everywhere.**
> Pooled over 3 seeds on `full` (`T = 100`), `D` reads PearlSCMRecourse **1.0**,
> CftsCels 6.6, NoiselessSCMRecourse 69.7, CftsCOMTE 83.3, CftsWachter 96.4, CftsConfeti
> **100.0** — every timestep declared an intervention, a rewrite rather than an
> explanation. CftsWachter fails both readings (`D = 96.4` **and**
> `Δ_total = 0.39`); `full_nl` reproduces the pattern. The positive control
> lands where it must: the single-`do()` method is the only one at `D = 1`.

**2. Intervention-to-outcome distance as a required reporting condition.**
SCM stability requires spectral radius < 1, so an intervention's effect on the
label attenuates geometrically in the distance from `t0` to the label site.
*Recourse feasibility is therefore a property of the benchmark configuration,
not of the method*, and beyond a computable horizon no causally valid recourse
exists — any validity reported there exploits a non-causal artifact,
demonstrated on the noiseless-rollout variant, whose validity survives an
intervention with measured effect ~1e-4.

The contribution is the **diagnostic and the reporting rule**, not the negative
result: many published temporal-recourse validity numbers are uninterpretable
without a stated intervention-to-outcome distance. Framing this as "an
impossibility" overstates it and understates the part that generalises — the
rule applies to every temporal recourse evaluation, including configurations
where recourse **is** feasible.

> **The distance is to the *label site*, not to `T` — H4 evidence (ii),
> confirmed 2026-08-03.** Every preset used to label terminally, so the evidence
> could not distinguish the readings (RISK-19). `smoke_interior_label` moves the
> label to `0.6 T` with SCM, `T`, noise and seed fixed: at `T − t0 = 2`,
> PearlSCMRecourse scores validity **1.00** with the label at `t = 29` and **0.10**
> at `t = 18`. Under a "decays in `T − t0`" reading those cells must agree.
> Interventions placed *after* the label site cannot move it, however close they
> sit to the end. Not confounded by classifier quality (interior-label LSTM 0.79
> vs `smoke` 0.92), because the decisive contrast is **within** the interior
> config: `t0 = 28` gives 0.10, `t0 = 18` gives 1.00.
>
> **Scale honesty (3 seeds).** At paper scale the effect is present but weak: no
> matched-`T − t0` cell separates, and label distance beats trajectory distance
> only on correlation (ρ = −0.99 vs −0.89). The ten-fold contrast is a
> *smoke-scale* result and must be labelled as one. This is why the rule is
> stated as `t_label − t0`: a benchmark reporting only `T − t0` reports the
> wrong distance whenever the label is not terminal, which is most applied
> settings.

> **Corroborated by a group that engineered around it rather than measuring
> it.** CausalTimePrior (App. B) reports the same attenuation from a *different*
> stability mechanism — no spectral radius constraint, but bounded activations,
> geometric lag decay (`p·0.7^k`) and clipping — concluding that *"causal
> effects attenuate rapidly — typically within 1–5 steps after intervention
> onset"* and that a model trained on that prior *"would inherit this
> short-range bias."* Their response was to **restrict** PFN queries to 0–5
> steps. Two independent designs, two stability routes, the same wall — which
> makes this a property of stable temporal SCMs, not an artifact of our VAR
> parameterisation. We **measure** the wall instead of avoiding it.

**3. A benchmark suite with exact structural counterfactuals.**
**LinearSCM-T** (VAR(L)) and **NlinearSCM-T** (per-node additive-noise MLP
transitions), each shipping true graph, true mechanism, and oracle
abduction–action–prediction counterfactuals. Mechanism-generic: any
additive-noise SCM plugs in unchanged. The label functional is a **configurable
axis** (`benchmarks/labels.py`), which is what makes the §4.2 reporting rule
testable rather than assumed.

**CF-faith is the admission gate, not a fourth contribution.** Two mutually
exclusive semantics (`noiseless_rollout`, `pearl_delta`) — a single CF cannot be
hard-faithful under both, and that contrast is itself a result — but it is a
pass/fail check with no dynamic range across real methods (§7). Listing it as a
contribution invites evaluation as a ranking metric and a finding of
degeneracy; it belongs inside §4.1 as the stage that admits or rejects before
`D` and PNS say anything.

# 5. Evaluation Protocol

**Three axes, and every reported metric belongs to exactly one** (fixed
2026-08-03; the map is `causaltemp_xai/metrics/taxonomy.py`, and a test fails if
a metric has no home):

| axis | scores | unit |
|---|---|---|
| **A — structure** | the benchmark | per **dataset** |
| **B — robustness** | methods | per method |
| **C — counterfactual quality** | methods | per method |

> **The asymmetry survives the renaming and must be preserved in the
> manuscript.** A characterises the data-generating process; B and C
> characterise explainers. Presenting all three as parallel per-method scores is
> a category error. Giving every metric a stated axis fixed a naming gap; it did
> not make A commensurable with B and C.

The former **Axis A (concept/representation quality)** was deleted 2026-08-03
with the concept-based methods it scored.

## Axis C — Counterfactual Quality *(per method)*

- **Validity** — fraction of CFs that flip the classifier. **Must be reported
  alongside the intervention-to-outcome distance**; per §4.2 it is not
  interpretable without it.
- **Proximity** — `‖x̂ⁿ − xⁿ‖_p` (L1 / L2).
- **Sparsity** — fraction of the `T × k` features left **unchanged**
  (`|Δ| ≤ INTERVENTION_TOL`); **higher = sparser edit**, `1.0` = CF identical to
  the original. Complement emitted as `frac_altered`. Structured variants
  `sparsity_channels`/`sparsity_timepoints` report the fraction of channels
  (resp. timesteps) left *entirely* untouched.
  *(Corrected 2026-08-11 under R5: this previously read "L0 fraction of altered
  features", the exact inverse of `metrics/axis_c.py:sparsity`, so every
  sparsity number read backwards against this specification.)*
- **OOD plausibility** — Isolation Forest, LOF.
- **CF-faith** *(admission gate)* — no-retroactive-change ∧ graph-propagation,
  under `noiseless_rollout` or `pearl_delta`. Returns **NaN**, not 1.0, when
  `intervention_t ≥ T−1` (the forward-simulation loop is empty, so the metric
  has no evidence either way); callers `nanmean` and publish `frac_degenerate` /
  `n_cf_faith_scorable` alongside every reported value.
- **TRSI** — Temporal Relevance Smoothness Index; mechanism-free descriptor of
  edit coherence. **Not** claimed as a contribution.

## Necessity / Sufficiency Audit — PNS *(per method)*

For a factual `x`, a method's CF `x_cf`, and `t0 = derive_intervention_t(x, x_cf)`
(the same intervention CF-faith scores):

| symbol | outcome fn | trajectory | meaning |
|---|---|---|---|
| `A` | classifier | method's `x_cf` | what the explanation **claims** |
| `B` | classifier | oracle CF of the same intervention | the claim, on the world's trajectory |
| `C` | SCM label rule | oracle CF of the same intervention | what the world **does** |

```
Δ_total      = A − C
Δ_trajectory = A − B     (the proposed CF is not what the world would produce)
Δ_outcome    = B − C     (the classifier does not track the true label rule)
Δ_total      = Δ_trajectory + Δ_outcome
```

Signed, both directions informative: **> 0** the explanation claims an efficacy
the world denies; **< 0** the model is blind to a real dependence. The oracle
uses **Pearl semantics** (re-injecting the abducted factual noise), because the
question is what would have happened *to this instance*.

**Do-complexity `D` is reported on every row.** The intervention is *extracted*
from `x_cf`, and the extraction is a modelling choice with two defensible
endpoints. `D` is what does not move between them, and a `Δ` of either kind must
never be cited without it. `D = 0` is the vacuous case (RISK-17), so
`frac_vacuous` is the `D = 0` row of the same distribution.

Both directions are estimated — **PS** on flip-candidates, **PN** on the
complement — and combined as `PNS = P(x,y)·PN + P(x',y')·PS`. PN, PS and the
mixing weights are **always reported separately**: on flip-candidates alone the
model side collapses into validity and the estimand is PS, so a combined-only
report would misname the quantity (R3).

## Axis B — Robustness *(per method)*

- **Shift-VR** — validity retention on held-out environments `E_test`.
- **Input sensitivity** — Lipschitz constant w.r.t. the input, where exposed.

## Axis A — Structure *(per dataset)*

SHD, lag accuracy, lagged-edge F1, residual dependence, graph-error
decomposition, AUC-ROC against ground truth — computed once per dataset in
Phase 01. It characterises the *benchmark*, not any method, which is why its
unit of analysis is recorded alongside the letter in `taxonomy.AXES`.

> **"Near-inert on the synthetic tiers" withdrawn 2026-08-12.** That was a
> property of the pre-P0-2 degenerate mechanism, not of the axis: post-retune a
> random graph costs **0.748** and DYNOTEARS's own well-recovered graph
> **0.163–0.687** (including 0.547 at AUC 0.965). The diagnostic is **live**. It
> still must not be printed as a per-method score alongside B and C, but on the
> unit-of-analysis argument alone.

> **M4f's instability caution was withdrawn 2026-08-12, on regeneration.** The
> sweep fits DYNOTEARS once per config/seed; a bootstrap ensemble (B refits on
> resamples of the dataset's own trajectories) tests whether that fit is
> representative. Pre-P0-2 (`smoke_nl` seed 0, B=5) it looked alarming:
> `mean_pairwise_shd = 4.0` against `true_edges = 7` — two refits disagreeing on
> more than half the true edge count — with the ensemble mean **0.00385**
> [0.00229, 0.00589] sitting *above* the single fit's 0.000131.
>
> Re-run on the retuned substrate, same code, same config and seed: **every one
> of the 5 members recovers an identical graph** (`mean_pairwise_shd = 0.00`,
> ensemble `graph_error` 0.06122 with a zero-width CI, exactly equal to the
> single main fit at SHD=4). The apparent instability was the degenerate
> mechanism again: with the true graph carrying 0.37% of the variance DYNOTEARS
> was fitting noise, so resamples disagreed. Once the graph drives 91% of the
> variance it is recovered consistently. **One seed, smoke scale** — a tight
> ensemble still cannot rule out shared bias (§6, M4g), so this licenses no
> claim that the recovered graph is *correct*, only that it is stable.

# 6. Benchmarks

> **What the tiers are for.** They are **external validity** for the findings,
> not validation of the metrics. A metric can only be shown faithful where a
> ground-truth mechanism exists to check it against — tier 1 and nowhere else.
> Tiers 2 and 3 have no oracle, so by construction they cannot tell you whether
> CF-faith, PNS or do-complexity measure what they claim; they test whether the
> tier-1 phenomenon travels. Presenting the tiers as a realism ladder that
> progressively strengthens the claim is the error this note prevents:
> increasing realism buys generality and nothing about construct validity.

**Tier 1 — Synthetic, ground-truth graph.** The only tier where the causal claim
can be checked against truth, and therefore where every headline result **and
every metric-validity argument** is established.

- **LinearSCM-T** — VAR(L). `k ∈ {5,10,20}`, `L ∈ {1,3}`, sparsity
  `s/k² ∈ {0.1,0.2,0.3}`, non-Gaussian noise (Laplace, uniform), `S ∈ {5,10}`
  segments, `T ∈ {50,100,200}`, `N = 10 000`.
- **NlinearSCM-T** — same structure, per-node MLP transitions:
  `x_t^i = decay_i·x_{t-1}^i + gain·tanh(MLP_i(masked lags)) + ε_t^i`,
  spectral-norm-capped. **Additive noise is load-bearing**: it makes Pearl
  abduction an exact subtraction, so CF-faith, the oracle structural CF, and
  exact PNS carry over unchanged from linear to nonlinear.
- **SpringSCM-T** (adopted 2026-07-31, M4c) — 2nd-order Newtonian from Kipf et
  al.'s NRI benchmark, as used by Bahri et al.:
  `v_{t+1} = v_t + m Σ_j E_ij[−k(r_t^i − r_t^j)] + η`, label = threshold on
  particle 1's terminal energy. Additive-noise, so the protocol carries over.

  > **It exists to attack our own horizon claim.** LinearSCM-T and NlinearSCM-T
  > both *enforce* contraction (spectral radius < 1), which is exactly where the
  > §4.2 wall comes from — so a reviewer can fairly ask whether H4 is a property
  > of temporal SCMs or of our stabilisation. Springs are conservative
  > (measured ρ = +0.024 vs `full_nl`'s −0.689) yet still carry terminal-time
  > labels, putting H4 in the regime where effects **persist**.

  > **KuramotoSCM-T removed 2026-08-11.** The three remaining synthetic families
  > are sufficient. H5 evidence (ii) no longer has a physically-motivated
  > non-monotonic mechanism and rests on the synthetic `tanh`→`sin` ablation
  > alone, which remains an open xfail pending an activation-choice decision.

**Discovery-method comparison across families (M4i, 2026-08-06).**
`experiments/09_tier1_synthetic_suite.py` runs DYNOTEARS and PCMCIplus's
cross-method agreement check (AUC-vs-true, SHD-vs-true, SHD-between-methods) on
all families including LinearSCM-T — previously excluded by a guard that was
never load-bearing for that check (it compares raw adjacencies, never masks a
mechanism). The deeper CF-faith graph-error decomposition stays restricted to
`mlp`/`spring`: `LinearMechanism` genuinely has no masking support, a real
limitation rather than an oversight.

**Tier 2 — Real signal, horizon external validity** (M4b, repurposed
2026-07-31). A UCR/UEA multivariate subset used for **one** purpose: testing
whether the §4.2 horizon result holds outside synthetic SCMs. Requires a
classifier, a label site, and a measurable intervention-to-outcome distance —
**no causal graph**. Criteria: `k ≥ 3` channels, genuine classification task,
moderate length, no missingness requiring imputation.

> **Discovered-graph CF-faith, reopened narrowly (M4g, 2026-08-06).** The
> 2026-07-31 decision descoped discovered-graph CF-faith on real data, reasoning
> that a number computed against a *discovered* graph cannot be distinguished
> from the graph being wrong in the same direction as the CF — a claim about
> **bias**. M4f's bootstrap ensemble resolves the *other* half: it quantifies
> **estimation variance** and reports it alongside every
> `cf_faith_discovered_*` number. **It does not resolve the bias half.** Every
> member shares DYNOTEARS's linear structural-equation assumption; if that is
> wrong for `BasicMotions` (motion-capture, plausibly nonlinear), all members
> can agree tightly while being wrong together. M4h narrows this only at the
> *graph* level (SHD between two structurally different methods), not the
> *mechanism* level these numbers depend on — PCMCIplus has no rollout-usable
> mechanism. **Variance quantified, bias disclosed, not resolved.** First
> measured (`BasicMotions`, B=5): `cf_faith_discovered_rollout_mean` ranges
> 0.081 (CftsConfeti) to 0.469 (CftsCounts) across the five graph-free methods.

Mechanism-level discovered-graph CF-faith otherwise stays out of scope; M4g is a
narrow disclosed exception, not a reversal.

> **Graph-level cross-check (M4i).** `run_cross_method_agreement_real` fits
> DYNOTEARS and PCMCIplus once each on Tier-2 data and reports
> `shd_between_methods` — structural agreement only, no correctness claim, since
> real data has no true graph. Materially weaker than Tier 1's AUC-vs-truth
> comparison and labelled as such in every file it writes.

**Tier 3 — SepsisSim** (deferred; M5 go/no-go). MIMIC-IV sepsis
cohort with a clinician-validated DAG, dynamic treatments, time-varying
confounding. If green, this is the only tier where discovery is **load-bearing**
— a graph is discovered or domain-supplied and propagated to both methods and
metrics, unlike Tiers 1/2 where discovery runs only as a diagnostic. Its
Axis-B error is then the ceiling on everything scored against it, reported as a
limitation.

> **Generic scaffold (M4i), not a commitment to SepsisSim.**
> `experiments/11_tier3_real_suite.py` is a dataset- and graph-source-agnostic
> orchestrator (`known`/`discover`/`domain`) built ahead of M5, exercised only
> against wired Tier-2 datasets and labelled a machinery demonstration. Building
> it is not a Go/No-Go decision.

# 7. Why CF-faith Is a Gate and `Δ_total` Is the Ranking Axis

CF-faith (hard) is an **indicator**, and every method that is not a positive
control fails it identically. On `full`, post-`d3c377d`, the column reads NoiselessSCMRecourse
1.00, PearlSCMRecourse NaN, every other method exactly 0.00 — and NoiselessSCMRecourse's 1.00 is
tautological, since it emits the noiseless rollout it is scored against. A
ranking statistic over that column measures "is this the oracle", not method
quality; more seeds and methods produce a longer column of zeros with tighter
CIs, not a finding.

**PNS ranks at `smoke` scale and not at `full` — measured, not assumed.** At
`T = 30` it separates the field (PearlSCMRecourse 0.90 vs 0.10–0.15). At `T = 100`,
both directions run (3 seeds, n_cf = 50), it separates nothing: **`PN_world` is
0.00 [0.00, 0.00] for all six methods**, `PS_world` is 0.01–0.02 with every CI
touching zero, combined PNS 0.00–0.01.

The collapse is **structural, not a defect**. PN conditions on instances the
classifier already places in the target class and asks whether the intervention
removes them; at `t0 ∈ {25, 50}` before a terminal label at `T = 100` nothing
moves the outcome in either direction. Both populations were baselined: the
unintervened world-label rate is 0.00 in both directions, so `C = 0` is exactly
zero causal effect, not a small one.

**What ranks at paper scale is `Δ_total`**, per direction, read with `D`:

| method | `Δ_total` PN | `Δ_total` PS | `D` |
|---|---|---|---|
| PearlSCMRecourse | **0.00** | 0.01 | **1.0** |
| CftsCels | 0.83 | 0.76 | 7.4 |
| NoiselessSCMRecourse | **1.00** | **0.21** | 75.0 |
| CftsWachter | 0.94 | 0.91 | 99.7 |
| CftsCOMTE | 1.00 | 0.99 | 85.5 |
| CftsConfeti | 1.00 | 0.98 | 54.0 |

> **Report PN and PS separately, always.** On one seed the combined figure reads
> `PNS = +0.01 = 0.50·PN(0.00) + 0.50·PS(0.02)`. Cited alone that looks like a
> joint necessity-and-sufficiency quantity; it is entirely the sufficiency term.
> The R3 naming hazard, observed on real data rather than argued in the
> abstract.
>
> **NoiselessSCMRecourse is direction-asymmetric on `full`, seed-unstable on `full_nl`.** On
> `full`, `Δ_total` PN = 1.00 on every seed against PS 0.00/0.62/0.00 — a stable
> asymmetry, the only large PN/PS gap in the set. On `full_nl` the direction does
> not hold: seeds 0 and 1 give PN 0.00 / PS 1.00, seed 2 gives PN 1.00 /
> PS −0.02, so the pooled mean describes no seed that was run. Two earlier
> readings are withdrawn — a mechanistic explanation, and a claimed stable
> "inversion" on `full_nl` that was a one-seed artifact. What stands: real on
> `full`, unstable on `full_nl`, unexplained. A direction-averaged number would
> hide all of it.

**Therefore:** CF-faith admits or rejects; **`Δ_total` and do-complexity rank**;
PNS is reportable where the horizon leaves it non-degenerate (smoke) and
reported as a null with both terms shown where it does not (full); and the
intervention-to-label distance conditions the interpretation of all of them.

# 8. Methods Evaluated

| Family | Methods |
|---|---|
| Counterfactual | Wachter, COMTE, CONFETI, CELS (vendored `cfts` repo) |
| Causal recourse | NoiselessSCMRecourse (noiseless rollout), PearlSCMRecourse (noise-reinjecting) |
| SCM-regularised | `TSCausalCF` (was `CausalFeasibilityCF`, renamed 2026-08-06; Bahri et al. — the primary Related Work foil) |

Target ≥ 8 CF explainers spanning instance substitution, evolutionary/heuristic,
deep latent/generative, and causal recourse (M3). NoiselessSCMRecourse and PearlSCMRecourse are
**positive controls**, labelled as such wherever they appear. `TSCausalCF` is
**not** — unlike them, its faithfulness under this benchmark's CF-faith metric
is not true by construction; it is the third-party foil being critiqued (§3),
wired into the default full-scale method set 2026-08-06.

**Classifier held fixed:** LSTM. TCN and Transformer descoped 2026-07-08;
classifier generalisation is not the claim, and the two-mechanism-family axis
(linear VAR vs nonlinear MLP) is the stated generalisation evidence.

# 9. Hypotheses

Full pre-registration, thresholds and verdict standards live in the external
planning documents. Short form only here. Renumbered 2026-08-06 from 10
entries to **6**, ordered by strength; no verdict changed by the renumbering.
Closed entries predating that date use the old IDs (H3a, H8, H8b, H8c, H9) and
are left as historical record.

- **H1** — Standard temporal CF methods achieve high validity (> 0.9) with
  CF-faith ≈ 0. *Split verdict (2026-08-04):* the CF-faith clause is universal —
  all four wired standard explainers score exactly 0.000 [0.000, 0.000] on
  `full`. Validity > 0.9 is **not** — only CftsCOMTE (1.000) and CftsWachter
  (0.920) clear it; CftsCels (0.507) and CftsConfeti (0.527) do not. What holds
  regardless: none buy causal faithfulness with whatever validity they achieve.
- **H2** (was H3a) — Given a well-recovered graph, the oracle structural CF
  retains near-full CF-faith, so explainers' failures are **propagation**
  failures, not graph-estimation failures. ***REFUTED 2026-08-12.*** The earlier
  confirmation (graph error ≤ 0.004 even for a random graph) rested on the
  pre-P0-2 degenerate mechanism. Re-run: a random graph costs **0.748**, and
  DYNOTEARS's own graph costs **0.547 at AUC 0.965 / SHD 4** — the antecedent
  fails on its own terms, so the propagation-not-estimation inference no longer
  follows. Graph estimation is a first-order error source. H3 is unaffected (it
  never routes through an inferred graph), but "graph quality is not the
  problem" must come out of the manuscript.
- **H2′** (post-hoc 2026-08-12, **not pre-registered** — formulated on the data
  it is tested on, so exploratory until it reproduces on held-out seeds) —
  Explainer choice spans a wider range of mechanism-infidelity than graph
  quality does, including at chance-level corruption, in **both** contraction
  regimes: 1.879 σ (worst explainer) vs 1.119 σ (chance graph) on `full_nl`
  = 1.68×; 0.724 vs 0.082 on `smoke_spring` = 8.8×. Family-independence is what
  makes it worth stating — it compares the quantity M4e's withdrawn verdict
  wrongly called regime-dependent. A claim about *dynamic range only*: it does
  **not** assert graph error is negligible (a realistic DYNOTEARS graph costs
  0.556 σ on `full_nl`, above that family's median explainer at 0.457).
- **H3** (was H9, the lead empirical claim) — For every wired CF method,
  `Δ_total > 0`, attributable almost entirely to `Δ_trajectory`: the proposed CF
  is not what the world would produce, and when the world realises the method's
  own intervention the classifier does not flip. See §4 and §7.
- **H4** (was H8, two designed sub-tests folded in as its own evidence) —
  *Recourse validity decays with the intervention-to-outcome distance at a rate
  governed by the SCM's spectral radius; beyond a computable horizon no valid
  causal recourse exists, and any validity there is a non-causal artifact of the
  CF semantics.* Evidence base is the `t0` sweep (validity 0.00 at
  `t0/T ∈ {0.25, 0.5, 0.75, 0.9}`, recovering to 0.60 only at 0.95, required
  `max|δ|` growing 8.3e-07 → 4.7). **Evidence (i)** (was H8b): non-dissipative
  families show no such decay — confirmed, validity *increases* with horizon
  instead, ruling out "artifact of our own stabilisation". **Evidence (ii)** (was
  H8c): validity and PN decay with distance to the **label site**, not with
  trajectory length — confirmed at smoke scale, weakly supported at paper scale
  (direction only). While every preset labels at `T − 1` the two readings are
  observationally identical, so H4's main evidence alone is not falsifiable
  without evidence (ii) (RISK-19).
- **H5** (retrospective synthesis over three independently pre-registered and
  individually refuted ablations, was H5/H6/H7) — The validity–CF-faith
  divergence is **invariant** to noise family (Gaussian vs Laplace), mechanism
  monotonicity (tanh vs sin), and regime stationarity (a mid-sequence structural
  break) — a property of the additive-noise causal structure itself, not of any
  one generalisation axis. Each original prediction was refuted as literally
  worded (M4); this synthesises what the refutations converged on and was
  **not itself pre-registered**.
- **H6** (was H4, rank-correlation) — Rankings by traditional CF metrics do not
  track rankings by CF-faith. **Demoted 2026-07-31 to a descriptive
  observation**: underpowered (n = 3 methods, tied ranks) and structurally hard
  to power, since CF-faith has no dynamic range across real methods (§7). Must
  never again be labelled the key finding.

**Not a hypothesis — construction check.** The former H3b ("causal recourse
attains CF-faith > 0.7") is entailed by definition: NoiselessSCMRecourse emits the noiseless
rollout it is scored against, and PearlSCMRecourse's `Δ_trajectory = 0` in both
directions because the oracle is a Pearl rollout of the same intervention. This
verifies the pipeline. The non-tautological quantity for PearlSCMRecourse is `C` —
whether the world delivers the outcome change — and only `C` may be cited.

# 10. Scope Boundaries — What This Benchmark Does Not Address

- **Concept / representation quality** (former Axis A: ICC, MIG, DCI, MCC) —
  removed 2026-08-03, axis and code. Reinstating means reinstating an axis, not
  restoring a module.
- **Attribution-method evaluation** — no axis scores them.
- **Causal sufficiency is assumed**; unmeasured confounders are future work.
- **Nonlinear *mixing* `g(z)` and non-additive (location-scale) noise are
  deliberately deferred.** NlinearSCM-T covers nonlinear *transitions* with
  additive noise, which is what keeps abduction exact and therefore keeps
  CF-faith, the oracle CF and exact PNS valid. Relaxing additivity costs the
  exactness §3 identifies as this work's delta.
- **The horizon limit constrains the benchmark's own findings.** With spectral
  radius < 1, configurations exist in which *no* method can succeed, and
  `t0_fractions` specified as a fraction of `T` are scale-dependent in the wrong
  direction — values validated at `T = 30` are structurally hopeless at
  `T = 100`. Every validity and PNS number is reported with its `T − t0` **and**
  its `t_label − t0`; no cross-configuration comparison is meaningful without
  them.
- **The audit's answer depends on how the proposal is read.** Neither the
  single-slice nor the whole-schedule reading is privileged; `Δ_trajectory` is
  inflated by the first and zeroed by construction under the second. Only the
  pair `(D, Δ)` is reportable (RISK-18).
- **The label functional is a design choice, not a fact about the world.** `C`,
  `Δ_outcome`, PN and PS are all properties of it. Default is a terminal
  single-channel threshold; `benchmarks/labels.py` supplies alternatives, and H4
  evidence (ii) tests whether the horizon result survives moving the label site
  (RISK-19).
- **CF-faith has no dynamic range across real methods** (§7), which is why it is
  published as an admission gate. A benchmark whose gate rejects every entrant
  is reporting a finding, not a leaderboard.
- **`graph_error` is not scale-free.** It is a difference of `exp(−residual)`
  soft scores, so its sensitivity depends on the mechanism family's state
  magnitude. Cross-config comparison requires the `graph_error_sigma` companion
  (M4e re-derivation, 2026-08-12).
- **Tier 3 only:** causal-discovery accuracy is the ceiling on everything scored
  against the discovered graph.
