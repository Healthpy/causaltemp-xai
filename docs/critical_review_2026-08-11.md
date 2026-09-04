# Adversarial Peer Review — CausalTemp-XAI

**Reviewer:** Reviewer 2 (independent adversarial check)
**Date:** 2026-08-11
**Branch reviewed:** `contribution-refocus`
**Materials:** `CLAUDE.md`, `docs/general_plan.md`, `docs/05_evaluation_plan.md`,
`manuscript/paper.md`, `results/tables/*`, `results/tier1_suite/*`,
`results/*/dynotears/graph_error.json`, source under `causaltemp_xai/`.
**Independence note:** written without reading `docs/pi_reevaluation_2026-08-11.md`.
Every number below was re-derived from the repository, not taken on trust; the
diagnostics I ran are described inline so they can be reproduced or refuted.

---

## Recommendation

**REJECT** in the current framing, with an explicit invitation to resubmit a
substantially narrower paper.

The submission's defining asset is a benchmark whose ground-truth causal graph is
claimed to make counterfactual claims checkable against the data-generating process.
On the flagship nonlinear configuration (`full_nl`, the "nonlinear publication
result"), I measure that the ground-truth graph generates **1.06% of the variance of
the next-step conditional mean**, and the conditional mean is itself only **35.7%** of
the observed variance — so the causal graph the entire instrument is built around
accounts for roughly **0.4% of the data**. The remaining 98% of the mechanism is a
per-node decay coefficient that **is not an edge in the true graph at all**
(`graph[i,i,0] == 0` for all `i`). Once this is known, the paper's six headline
results — the near-zero `graph_error`, the inert Axis A, the null non-monotonic
ablation, the identically-zero PN, the exactly-zero CF-faith column, and the "H5
invariance" synthesis — stop being six independent findings about counterfactual
explanation methods and become six projections of one property of a badly
parameterised generator. A benchmark cannot report the absence of a signal it never
inserted as a finding about the methods that failed to find it.

Compounding this, the paper's own declared ranking axis (`general_plan.md` §7:
"What ranks at paper scale is `Δ_total`") is algebraically degenerate at both
endpoints of the bracket the authors themselves define (Major Concern 2), and the
headline statistics rest on a 3-cluster hierarchical bootstrap that the
pre-registration itself forbids. None of this is unfixable, and much of it is fixable
without new science — but it is not fixable in a revision cycle, and the current
claims cannot stand while it is fixed.

**I want to be equally clear about what is not wrong here.** This is one of the most
epistemically honest research artifacts I have reviewed. The authors retract their own
headline findings when the evidence moves (`Δ_trajectory ≈ 1.0`, the 2-seed
"disjoint intervals" reading, the CF-faith column at `d3c377d`); they record all three
H5 ablations as *refuted as literally worded* and refuse to reword the predictions;
they label construction checks as construction checks and forbid citing them as
evidence; they publish `NaN` rather than a flattering `1.0` when a metric has no
evidence; they report `frac_vacuous` beside the validity it undermines; and they state
in writing that their own real-data H4 test is *structurally unfalsifiable by their
design*. Section 6's boxed warning that "increasing realism buys generality, and buys
nothing at all about construct validity" is a better piece of methodological writing
than most published benchmark papers contain. The problems below are problems of
parameterisation and of claim calibration, not of integrity — and that is precisely
why they are worth stating at full force.

---

## Summary of the submission

The paper argues that temporal counterfactual-explanation (CF) evaluation measures
only the CF's relationship to the *model* (validity, proximity, sparsity, OOD
plausibility) and never its relationship to the *data-generating process*. It builds
synthetic additive-noise temporal SCMs with known graph and known mechanism, so that a
proposed CF can be scored against an exact structural counterfactual (abduction–
action–prediction, exact because additive noise makes abduction a subtraction). On top
of this it defines: (i) CF-faith, a mechanism-consistency admission gate under two
mutually exclusive semantics (`noiseless_rollout`, `pearl_delta`); (ii) a model-vs-world
audit decomposing `Δ_total = Δ_trajectory + Δ_outcome` with exact PN/PS; (iii)
do-complexity `D`, the number of timesteps a proposal must declare as interventions
before the mechanism can reproduce it; and (iv) a horizon reporting rule, that recourse
validity decays geometrically in the intervention-to-label distance so no validity
number is interpretable without it. Six CF methods (four third-party explainers, two
causal-recourse positive controls) are evaluated on linear and nonlinear synthetic
tiers plus a real UEA dataset, at 3 seeds.

The claimed headline: standard temporal CF methods achieve high validity with exactly
zero causal faithfulness, and "buy their claimed validity by intervening almost
everywhere" (`D` = 6.6 to 100 of 100 timesteps).

---

## Major concerns

### FATAL-1 — The flagship nonlinear benchmark's causal graph carries ~0.4% of the data, and every null result in the paper is a consequence of that fact

This is the concern that determines my recommendation and it subsumes evidence items
1, 3 and much of 5 in one mechanism.

`MLPMechanism.forward_numpy` computes
`mean_i = decay_i · x_{t-1}^i + gain · tanh(W2 · act(W1 · masked_lags + b1) + b2)`.
The `decay` term is a per-node autoregressive coefficient sampled from
`decay_range = [0.3, 0.8]`; the graph enters **only** through `masked_lags`. I
regenerated `full_nl` (`k=10, L=1, T=100, seed=42`, identical hyperparameters, `N=600`
for tractability — the mechanism is drawn before the data and is unaffected by `N`)
and measured, over 4,000 sampled history windows:

| quantity | measured value |
|---|---|
| true graph edges | 21 of 100 possible; **zero self-loops** |
| `std(decay term)` | 0.10503 |
| `std(MLP / graph-coupling term)` | 0.01090 |
| variance share of next-step mean: decay | **98.11%** |
| variance share of next-step mean: graph coupling | **1.06%** |
| mean `\|∂mean_i / ∂x_j\|` over true edges | 0.0369 |
| mean own-decay coefficient | 0.5255 (**14.3×** larger) |
| `var(conditional mean) / var(X)` | 0.357 |
| ⇒ share of observed variance flowing through the true graph | **≈ 0.38%** |

Everything the paper reports as a null now follows deterministically:

- **`graph_error ≈ 0.003` (evidence item 3, `general_plan.md` §5, H2's verdict).**
  Corrupting the graph can only perturb the masking of a branch that carries 1% of the
  conditional mean. `results/full_nl_seed0/dynotears/graph_error.json` records
  `cf_faith_inferred = 0.99817` at `corrupt_frac = 1` — a *fully rewired* graph leaves
  the oracle counterfactual reproducing to within 0.2%. That is not "graph quality is
  irrelevant when causal effects dissipate". That is "there was almost nothing flowing
  through the graph to disturb."
- **The non-monotonic ablation is a no-op (evidence item 1).** I confirmed the
  authors' own numbers and pushed further. Hidden pre-activations on `smoke_nl` /
  `full_nl` have mean `|z| = 0.0136 / 0.0152`, p99 `= 0.093 / 0.082`. Over that range
  `sin` and `tanh` are both indistinguishable from the identity: mean
  `|tanh(z) − z| / |z| = 1.8e-4`, and mean `|tanh(z) − sin(z)| = 5.2e-6`. Regenerating
  both datasets: `corr(X_tanh, X_sin) = 0.999999988`, and
  **`var(ΔX)/var(X) = 2.3e-8`**. The ablation perturbs two hundred-millionths of the
  data variance. `05_evaluation_plan.md` §5 H5 evidence (ii) states: *"Verified this is
  not a wiring bug … confirmed structurally different from the `smoke_nl` mechanism —
  the identical numbers are a genuine result, not a no-op ablation."* The wiring is
  indeed correct; the **conclusion is not**. Structural difference in the code is not
  the same as a manipulation the data can express, and the check performed
  (`_ACTIVATIONS_NP` is dispatched) tests the former while the claim requires the
  latter. Mechanism monotonicity was never varied in any way any downstream metric
  could see. H5 evidence (ii) is vacuous and must be withdrawn, not reported as
  invariance.
- **Is the whole "nonlinear tier" actually nonlinear?** I asked directly. I fit an
  unconstrained least-squares linear (VAR) model to the *true nonlinear mechanism's own
  conditional mean* on both `smoke_nl` and `full_nl`. Per-node **R² = 1.000000** (min
  across nodes 0.999999 on `smoke_nl`, 1.000000 on `full_nl`); relative L2 error of a
  global first-order Taylor surrogate is 3.2e-4 (`full_nl`) and 5.6e-4 (`smoke_nl`).
  `NlinearSCM-T` **is** a VAR to six decimal places. The `MLPMechanism` docstring
  asserts that the `1/√(n_parents)` input scaling makes "pre-activations land in tanh's
  curved region (genuine nonlinearity)"; the measurement says they land at |z| ≈ 0.015,
  which is the middle of the linear region. This invalidates the paper's stated
  generalisation evidence. `general_plan.md` §8 says: *"classifier generalisation is not
  the claim, and the two-mechanism-family axis (linear VAR vs nonlinear MLP) is the
  stated generalisation evidence."* There is no such axis. `full` and `full_nl` are two
  linear systems.
- **The two configs are not even the same kind of linear system, which makes the
  "carries over" claim worse, not better.** On `full` (`LinearSCM-T`) I measure
  `mean |a_ii| = 0.0000` (no self-loops in the coefficient matrix either) and
  `mean |a_ij| = 0.662` over the graph's edges, spectral radius 0.900 — i.e. on the
  linear tier **100% of the dynamics flow through the graph**. On `full_nl`, ~1% does.
  The two flagship configs sit at opposite ends of the only property that determines
  whether this benchmark can measure anything, and the paper treats them as a
  robustness axis.
- **The corruption sweep is only ever run where it cannot bite.** H2's verdict
  ("graph error ≤ 0.004 even for a fully random graph") is established **only on
  `full_nl`** — the 1%-coupling config. It is never run on `full`, the 100%-coupling
  config, because `general_plan.md` §6 records that `LinearMechanism` "genuinely has no
  masking support". So the single configuration on which H2 could be falsified is
  excluded by an implementation limitation, and H2 is confirmed on the one
  configuration where it is true by parameterisation. A reviewer does not need to
  allege cherry-picking here; the paper documents the exclusion itself.

**Fix.** Re-parameterise `NlinearSCM-T` so the graph-coupling branch is a
*designed, reported* fraction of the conditional mean (e.g. sweep coupling share ∈
{0.1, 0.3, 0.5, 0.8} by raising `gain`, lowering `decay_range`, or removing the
`1/√n_parents` input scaling until pre-activations reach `|z| ~ O(1)`), and report the
measured share as a first-class property of every config, in every table, next to
`T − t0`. Then re-run. Add a *positive* control for the graph-corruption sweep — a
config where `graph_error` is known to be large — so the near-zero reading is
interpretable as a measurement rather than as a floor. Add `LinearMechanism` masking so
H2 can be tested where it is falsifiable. Withdraw H5 evidence (ii) and re-run the
ablation with an activation change large enough to move the data (verify with
`var(ΔX)/var(X)` before scoring anything).

**Confidence a real reviewer raises this: HIGH.** It takes twenty lines of numpy and
any reviewer who asks "how much does your graph actually do?" will find it.

---

### FATAL-2 — `Δ_total`, the paper's declared ranking axis, is algebraically degenerate at both endpoints of the bracket the paper itself defines

`general_plan.md` §7 is explicit: *"CF-faith admits or rejects; **`Δ_total` and
do-complexity rank**"*. §4.1 and §10 are equally explicit that neither reading of a
proposal's intervention is privileged, so only the pair `(D, Δ)` is reportable. I
checked what `Δ_total` actually equals at each endpoint, from
`results/tables/table_pns_do_complexity_full{,_nl}.csv`:

**`schedule` mode — `B ≡ A` exactly, for all 6 methods, on both configs.**

| config | method | A | B | C | Δ_total | Δ_traj |
|---|---|---|---|---|---|---|
| full | CftsWachter | 0.920 | 0.920 | 0.533 | 0.387 | 0.000 |
| full | CftsCOMTE | 1.000 | 1.000 | 0.813 | 0.187 | 0.000 |
| full_nl | CftsCels | 1.000 | 1.000 | 0.952 | 0.048 | 0.000 |
| full_nl | CARLA | 0.667 | 0.667 | 0.667 | 0.000 | 0.000 |

`Δ_trajectory = A − B = 0` is not a measurement; it is an identity forced by the
reading (the schedule reading declares every edited slice an intervention, so the
oracle reproduces the proposal by construction — the paper says so). What remains,
`Δ_total = Δ_outcome = A − C`, is the disagreement between **the LSTM** and **the SCM's
label threshold rule** evaluated on the same trajectory. It contains no per-method
causal content whatsoever: it is a property of the classifier fit and of how far the
method's edit pushed the trajectory into a region where the two labellers disagree.
Ranking methods by it ranks them by classifier failure.

**`single_slice` mode — `B ≡ 0` and `C ≈ 0`, so `Δ_total ≡ A − ε`.**
Measured `Δ_total − A` across all methods and both configs: −0.0067, −0.0067, −0.0095,
−0.0185, −0.0067, −0.0068 (`full`) and −0.0067, −0.0067, −0.0533, −0.0185, −0.0067
(`full_nl`). `A` is the classifier's verdict on the proposed CF — i.e. **validity on
the PNS-selected subpopulation**. So under the single-slice reading, `Δ_total` is
validity minus a per-config constant of ~0.01.

**Therefore the paper's declared ranking axis is either (a) validity relabelled, or
(b) a classifier-fit statistic — and there is no third reading in between where the new
information lives.** The paper carefully brackets the two endpoints and then reports a
quantity that is uninformative at both. `B ≡ 0` in single-slice mode is itself the
FATAL-1 mechanism reappearing: the oracle realisation of a single-slice intervention
does not move the label because interventions in this benchmark do not move labels.

The same root cause explains §7's `PN_world = 0.00 [0.00, 0.00]` for all six methods
with a zero-width interval and an unintervened world-label baseline of 0.00. The paper
calls this collapse "structural, not a defect". It is structural — but the structure is
the generator's, not the field's, and a paper cannot claim to have measured a null
causal effect when its own generator supplies none to measure.

**Fix.** Either (i) find and defend a reading strictly between the two endpoints — e.g.
declare the top-`m` edited slices as interventions and report `Δ(m)` as a curve, with
`m = D` as one point — so that `Δ_trajectory` is a measured quantity rather than an
identity; or (ii) demote `Δ_total` to a diagnostic, promote `D` (which *is* a genuine
per-method quantity, invariant to the reading, and the paper's best result) to the
ranking axis, and re-title accordingly. Do not report `Δ_total` as a ranking axis
without showing, per config, that `B` is neither pinned at `A` nor at 0.

**Confidence: HIGH.** Both identities are visible by subtracting two columns in a
committed CSV.

---

### FATAL-3 — H5's "invariance" synthesis is built on three ablations, at least one of which does not exist as a manipulation

`05_evaluation_plan.md` §5 H5 states the divergence is invariant to noise family,
mechanism monotonicity, and regime stationarity, and calls this a property of the
additive-noise causal structure. Evidence (ii) does not manipulate monotonicity in any
measurable sense (FATAL-1: `var(ΔX)/var(X) = 2.3e-8`), so it contributes nothing to the
"convergence across all three independent sub-tests" the umbrella claim rests on. That
reduces H5 to two ablations, one of which (evidence (i), Gaussian vs Laplace at matched
scale) produces "CF-faith bit-for-bit identical" — a result that is itself entailed by
FATAL-1 plus the fact that CF-faith `hard` is a floating-point identity test (MAJOR-4):
a metric that returns exactly 0 or exactly 1 for structural reasons will be bit-for-bit
identical under *any* ablation, including ones that do work.

The paper is scrupulous about flagging that H5's umbrella was not pre-registered. It is
not scrupulous about the prior question: whether an invariance claim can be supported by
ablations whose effect size on the data was never measured.

**Fix.** Report `var(ΔX)/var(X)`, label-agreement, and a distributional distance between
ablated and base datasets as a **precondition gate** for every ablation, before any
downstream metric is computed. Any ablation failing the gate is reported as
"manipulation absent", not as invariance. Re-run evidence (ii) with an activation or
gain change that clears the gate. Recompute the H5 umbrella from whatever survives.

**Confidence: HIGH** once FATAL-1 is on the table; **MEDIUM** as a standalone catch.

---

### MAJOR-4 — CF-faith `hard` is a floating-point identity test against the oracle, and its R6 adversarial test cannot detect that

Evidence item 2 asked whether CF-faith restates method definitions. It does, and the
code says why. `causaltemp_xai/metrics/cf_faith.py:243`:

```python
l1_residual = float(np.abs(residual_region - simulated_region).mean())
hard = float(l1_residual < self.tol)      # self.tol = 1e-3
```

The threshold is a **mean absolute residual below 1e-3**, against data with
`std(X) = 0.176` (`full_nl`) and `0.355` (`full`) — i.e. 0.28% and 0.57% of one standard
deviation, averaged over the entire post-`t0` region. Nothing passes that except a
trajectory produced by calling the same `mechanism.forward_numpy` the metric calls.
CARLA emits exactly that; PearlCARLA emits exactly the Pearl variant; nothing else can.
`hard` is therefore an indicator of *oracle identity*, and the mutual exclusivity that
`CLAUDE.md` defends as "itself a benchmark result" is the statement that CARLA is not
PearlCARLA. That is true, and it is not a finding about counterfactual explanation.

**On R6 compliance.** `docs/axis_metrics_report.md` rows 63–67 mark CF-faith green.
Reading `tests/test_metric_adversarial.py::TestCFfaithAdversarial`, the suite tests
exactly two states: an oracle CF (assert `hard == 1`) and a blatant violator — a
retroactive `+5.0` edit, or `cf[t0:] += N(0, 0.5)` (assert `hard == 0`). **There is no
test case between those poles.** No test constructs a partially-mechanism-consistent
CF and asserts the metric orders it correctly. So the R6 gate certifies that the metric
correctly identifies the oracle; it does not — and cannot, as written — establish that
it measures a graded property of methods. R6 is satisfied to the letter and fails at
its stated purpose ("construct a CF that *should* fail the metric; verify rejection"
has been read as "construct a CF that should obviously fail").

**And the paper discards the column that does discriminate.** `soft = exp(−residual/scale)`
is monotone in the same residual, and the committed tables show it has real, tight,
largely non-overlapping dynamic range:

| config | CARLA | CftsCels | CftsCOMTE | CftsConfeti | CftsWachter | PearlCARLA |
|---|---|---|---|---|---|---|
| `full` soft-rollout | 0.9999 | 0.8635 [.844,.882] | 0.6751 [.612,.719] | 0.8197 [.805,.835] | 0.8055 [.789,.823] | 0.8115 |
| `full` soft-pearl | 0.8125 | 0.9937 [.992,.995] | 0.7637 [.688,.815] | 0.8594 [.844,.875] | 0.7743 [.755,.791] | 0.9999 |

`general_plan.md` §7 and §10 assert "CF-faith has no dynamic range across real
methods", and `05_evaluation_plan.md` RQ6 demotes H6 (rank correlation) on exactly that
ground — "more seeds and methods yield a longer column of zeros". That justification is
false of the soft column the project computes, publishes, and plots (`fig1_scatter` uses
`cf_faith_rollout_soft`). The degeneracy is manufactured by a binarisation choice at
`tol = 1e-3`, and the paper's central methodological narrative ("a gate, not a
leaderboard") is then derived from the degeneracy it created.

**Two factual errors in the write-up, R5 violations.** §7 states: *"On `full`,
post-`d3c377d`, the column reads CARLA 1.00, PearlCARLA NaN, and every other method
exactly 0.00."* `table_seed_aggregate_full_lstm.csv` says PearlCARLA
`cf_faith_rollout_hard = 0.0` with `n = 141` (not NaN — that is `full_nl`), and
CftsCels `cf_faith_pearl_hard = 0.192 [0.114, 0.276]` with `n = 99` (not 0.00). The
task brief I was given repeated the same "all four score 0.0/0.0" claim; it is wrong,
and the counterexample slightly weakens the tautology charge while being an independent
documentation defect.

**Fix.** (a) Replace `hard` with the residual itself, reported on a log scale with the
oracle at the floor and a stated, defended tolerance sweep — or report `hard` at several
tolerances so the reader sees the cliff. (b) Add R6 test cases at intermediate
faithfulness (e.g. an oracle rollout with the mechanism's coefficients perturbed by
ε ∈ {1%, 10%, 50%}) asserting monotone `soft` and a `hard` cliff at a *stated* ε; until
those exist, R6 is not met for this metric in the sense R6 intends. (c) Correct §7
against the committed tables in the same commit (R5).

**Confidence: HIGH** for the tautology framing; **HIGH** for the R5 errors.

---

### MAJOR-5 — A reported method is degenerate and its degeneracy is what validates the instrument

Evidence item 4, verified. `PearlCARLA` on `full_nl`: `validity = 0.0 [0.0, 0.0]`,
`proximity_l1 = 1.40e-05`, `sparsity = sparsity_channels = sparsity_timepoints = 1.0`
(no cell altered), all four `cf_faith_*` columns `NaN` with `n = 0`, and every column of
`table_pns_do_complexity_full_nl.csv` `NaN`. On `full` it is barely better:
`validity = 0.02`, `proximity_l1 = 2.68`.

The honesty is correct — `NaN` with `n = 0` is the right output. The problem is what the
paper does with the `full` row. `general_plan.md` §4.1 cites *"the positive control
lands where it must: the single-`do()` recourse method is the only one at `D = 1`"*, and
§7's ranking table cites PearlCARLA `Δ_total` PN = 0.00 / PS = 0.01, `D` = 1.0 as the
instrument's validation. But `D = 1` and `Δ ≈ 0` are also exactly what **a method that
edits essentially nothing** produces — and on `full` PearlCARLA's `validity = 0.02` and
`proximity_l1 = 2.68` (against CARLA's 146.7 and Wachter's 258.4) say it is very close
to that. The positive control that validates the audit instrument is, on the flagship
configs, nearly indistinguishable from a no-op. The paper's own `is_vacuous_intervention`
detector exists precisely to catch this class of failure (RISK-17) and is not applied to
the control in the tables that cite it.

**Fix.** Either report a `full_nl` result for PearlCARLA or remove it from every
`full_nl` table and state why in the caption — an all-`NaN` row invites "did the method
fail or did the harness?" and the paper does not answer. Report `frac_vacuous` for
PearlCARLA in the same table as its `D`. Add a **negative control**: a literal no-op CF
scored through the whole pipeline. If the no-op's `(D, Δ_total, PN, PS)` is not clearly
separated from PearlCARLA's, the instrument has not been shown to distinguish "clean
single intervention" from "did nothing", and contribution 1 is not established.

**Confidence: HIGH.** The `NaN`-with-`n=0` row is the first thing a reviewer's eye finds
in a results table.

---

### MAJOR-6 — Degenerate causal-discovery outputs are reported as measurements of method suitability

Evidence item 5, verified against `results/tier1_suite/table_method_suitability.csv`
and `summary.json`'s `per_seed_records`:

- **Kuramoto:** `dynotears_auc = 0.5` and `pcmciplus_auc = 0.5`, **exactly**, on all
  three seeds independently, with `shd_between_methods = 0.0` on all three. Two
  structurally unrelated discovery algorithms returning byte-identical adjacencies at
  exactly chance AUC across three independent datasets is the signature of a constant
  (almost certainly empty) adjacency, not of a measurement. `shd_to_true` is 10/4/10 —
  which, against `true_edges = 5` (from `smoke_kuramoto/dynotears/graph_error.json`), is
  consistent with "every true edge missed, no false positives".
- **Spring:** `dynotears_auc = 0.460` pooled (0.458 / 0.464 / 0.458 per seed) —
  **below chance**, with a range of 0.006 across seeds. `axis_a` for
  `smoke_spring_seed0` reads `LagAcc = 0.0`, `LagF1 = 0.0`, `AUC = 0.4579`.

A below-chance AUC with near-zero variance across seeds is a sign convention error or a
degenerate fit, not a finding about DYNOTEARS. The table is titled "method suitability"
and will be read as "PCMCIplus is more suitable than DYNOTEARS on spring (0.664 vs
0.460)"; that comparison is not supported by these numbers.

**Fix.** Print the inferred adjacency's edge count and density beside every AUC. Add an
assertion that rejects a constant adjacency rather than scoring it. Investigate the
below-chance spring AUC as a probable orientation/sign bug before publishing the
comparison. Until then, mark both rows "discovery failed" rather than reporting a
number.

**Confidence: HIGH.** `AUC = 0.500` with zero variance is a red flag any methods
reviewer reacts to immediately.

---

### MAJOR-7 — The graph-quality "dynamic range" comparison is confounded by scale, and the correct matched-scale control is in the repository, unused

`general_plan.md` §3 claims spring/kuramoto give "**45×–123×** the dynamic range" of
`full_nl`'s `graph_error`, and concludes: *"Graph quality matters when causal effects
persist, and is close to irrelevant when they decay."*

The comparison is `full_nl` (`k=10, T=100, N=10,000`) against `smoke_spring` /
`smoke_kuramoto` (`T=30, N=500`). Three of the four things that could explain the
difference vary simultaneously: horizon, sample size, and mechanism family. The
matched-scale control exists — `results/smoke_nl/dynotears/graph_error.json`, the same
dissipative MLP family at `T=30` — and it reads `graph_error = 0.01618` at
`corrupt_frac = 1`, **5× larger than `full_nl`'s 0.00325**. Recomputing the claimed
ratio at matched scale:

| comparison | ratio |
|---|---|
| spring / `full_nl` (**as published**) | 123× |
| kuramoto / `full_nl` (**as published**) | 45× |
| spring / `smoke_nl` (**matched T, N**) | **24.7×** |
| kuramoto / `smoke_nl` (**matched T, N**) | **9.0×** |

The published ratio is inflated roughly fivefold by a horizon/sample-size difference the
sentence attributes entirely to dissipation. And FATAL-1 supplies the real explanation,
which is neither: `graph_error` scales with the fraction of the conditional mean that
flows through graph edges — 1% for MLP, ~100% for spring/kuramoto, where the coupling
*is* the dynamics.

**Two further defects in the same sweep.** (a) The per-seed spring curve is
**non-monotone**: `smoke_spring/dynotears/graph_error.json` gives graph_error
0 → 0.400 → 0.190 → 0.236 → 0.484 across `corrupt_frac` 0 → 1, so the response is
*lower* at 50% corruption than at 25%. H2's own evidence standard requires "a
graph-quality sweep across a true→random ladder showing **monotone** `graph_error`
response". The pooled 3-seed table is monotone; the per-seed curve backing it is not,
and only the pooled version is shown. `smoke_nl`'s ladder is likewise non-monotone at
the top (0.01587 → 0.01698 → 0.01618). (b) `smoke_spring` and `smoke_kuramoto` saturate
at `SHD = 8–10` (5 true edges) while `full_nl` reaches `SHD = 35.3` (15–21 true edges) —
the x-axes are not comparable either.

**Fix.** Replace the cross-scale comparison with `smoke_nl` vs `smoke_spring` vs
`smoke_kuramoto` at matched `T`, `N`, and comparable `SHD` normalisation
(`SHD / true_edges`). Plot per-seed curves, not only the pooled mean. Re-attribute the
finding to measured coupling share, and state the ratio at matched scale.

**Confidence: HIGH** for the confound; **MEDIUM-HIGH** for the monotonicity failure
(a reviewer has to open the per-seed JSON, but the H2 evidence standard invites it).

---

### MAJOR-8 — The statistical protocol violates the paper's own pre-registration on the flagship configs, and the CIs are not usable

`05_evaluation_plan.md` §6 requires **seeds ≥ 5** and **`n_cf` ≥ 100 (full)**. Every
headline table is 3 seeds × `n_cf = 50` (`validity_n = 150`). Both requirements are
missed on both flagship configs. The paper flags the seed shortfall repeatedly and
honestly; it does not flag the `n_cf` shortfall with equal prominence, and neither
shortfall is treated as blocking.

More seriously, `aggregate_across_seeds` runs a **hierarchical (cluster) bootstrap over
seed clusters** — methodologically the correct choice, and I credit it — but with
**G = 3 clusters**. The cluster bootstrap's sampling distribution over 3 clusters has
10 distinct multisets; `n_boot = 10,000` resamples of it is arithmetic, not inference.
The output is visible in the tables: `CARLA` on `full_nl` reports
`validity = 0.667, CI = [0.000, 1.000]` — a 95% interval covering the entire parameter
space — and on `full`, `0.213, CI = [0.000, 0.633]`. Standard guidance for cluster
bootstrap is on the order of G ≥ 30–50; the pre-registered "≥ 5" was already
optimistic, and 3 is below any threshold at which the machinery reports information.
Any hypothesis "confirmed at the CI level" (§4's stated standard) on these tables is
confirmed against an interval that cannot exclude anything.

The one place this matters most: H1's CF-faith clause is reported as
`0.000 [0.000, 0.000]` and described as excluding 0.3 "by the widest margin the metric
can express". A zero-width CI over 3 clusters of an indicator that is structurally
pinned at 0 (MAJOR-4) is not evidence of precision.

**Fix.** Run ≥ 20 seeds on `full` and `full_nl` before any CI is reported — the paper's
own M4f timing evidence (~6 s for a full DYNOTEARS ensemble run) suggests cost is not
uniformly the blocker it is assumed to be, though the Phase 03 CF runs plainly are. If
compute forbids it, report per-seed values in a strip plot and **no CI at all**, and
state that hypothesis verdicts are deferred. Do not report a `[0.000, 1.000]` interval
as a result.

**Confidence: HIGH.**

---

### MAJOR-9 — CF-faith's denominator is chosen by the method being evaluated (differential attrition)

CF-faith returns `NaN` when `intervention_t ≥ T−1`, and `intervention_t` is derived from
the method's own output via `derive_intervention_t` (first `t` with per-element
`max|Δ| > 1e-3`). The scorable sample size therefore varies by method:

| config | CARLA | CftsWachter | CftsCOMTE | CftsCels | CftsConfeti | PearlCARLA |
|---|---|---|---|---|---|---|
| `full` (`cf_faith_*_n` / 150) | 150 | 150 | 150 | **99** | **79** | 141 |
| `full_nl` | 150 | 150 | 149 | **82** | **74** | **0** |

Roughly **half** of CftsConfeti's counterfactuals are dropped before scoring, and the
surviving half is selected on a property (where the method's first supra-tolerance edit
lands) that is plausibly correlated with faithfulness. The paper publishes `n` and
`frac_degenerate` — genuinely good practice, and a real safeguard — but nowhere treats
the resulting comparison as what it is: **a between-method comparison on
method-dependent subsamples**. A "0.000 [0.000,0.000]" computed on the 74 instances a
method did not degenerate on is not the same estimand as one computed on all 150.

Compounding it, `derive_intervention_t` also sets `T − t0`, the axis of contribution 2.
`05_evaluation_plan.md`'s M4b verdict identifies exactly this problem on real data
("the unconstrained-edit confound"; `CftsWachter`'s median derived `t0` is **0** on
`full` too, with `sparsity = 0.02`, i.e. 98% of cells edited) and correctly declares H4
untestable there. The same confound applies to the synthetic headline numbers — where
`CftsWachter` and `CftsConfeti` are both assigned `T − t0 = 100` by construction — and
is not carried across.

**Fix.** Report every CF-faith mean twice: on the scorable subset and with degenerate
instances imputed at 0 (the conservative reading), and show both. Run a sensitivity
analysis of `derive_intervention_t`'s tolerance (1e-3 is 0.3–0.6% of a data SD; sweep
1e-4 … 1e-1) and show that `t0`, `D`, and `frac_degenerate` are stable to it — if they
are not, the horizon axis and the CF-faith denominator are both artifacts of a
threshold. Propagate M4b's own confound finding to the synthetic tables.

**Confidence: MEDIUM-HIGH.** A careful methods reviewer will notice the varying `n`
column; a hurried one will not.

---

### FIXABLE-10 — Provenance: the committed graph-quality table cannot be regenerated from the committed artifacts

`run_graph_quality_report` (`experiments/08_aggregate_and_report.py`) reads
`results/<config>_seed<s>/dynotears/graph_error.json` and iterates
`payload["graph_quality_sweep"]`. In the current tree,
`results/smoke_{spring,kuramoto}_seed{0,1,2}/dynotears/graph_error.json` all have
`graph_quality_sweep: null` and `git_dirty: true`; only the unseeded
`results/smoke_{spring,kuramoto}/dynotears/graph_error.json` carry a sweep. So the
committed `table_graph_quality_full_nl_vs_smoke_spring_vs_smoke_kuramoto.csv`, which
reports `n_seeds = 3` for those configs, **cannot be reproduced from the current
artifacts** — the pooling code would raise on `null`. `results/tier1_suite/summary.json`
is likewise `git_dirty: true`.

This violates `05_evaluation_plan.md` §4's own reproducibility check ("Before any result
enters the manuscript: clean `git status`; … re-running the same script on the same
commit reproduces the output") and undercuts R10. The paper's most-cited external-validity
table is the one that fails it.

**Fix.** Re-run and re-stamp the six seeded spring/kuramoto sweeps clean, then
regenerate the table, then verify byte-equality. Add a CI check that every published
table is regenerable from committed artifacts.

**Confidence: MEDIUM.** Reviewers rarely check; artifact evaluators at NeurIPS/ICLR do,
and this repository invites artifact evaluation.

---

## The single most damaging question the authors currently cannot answer

> **"What fraction of your benchmark's observed data variance is actually generated by
> the ground-truth causal graph — the object your entire instrument is built to
> exploit — and what does that number do to every null result in your paper?"**

On `full_nl` the answer is **≈ 0.4%** (1.06% of the conditional mean, which is 35.7% of
the observed variance), with 98.1% of the mechanism carried by per-node decay
coefficients that are not edges in the graph. This number appears nowhere in the
repository: not in `axis_a_benchmark.json`, not in `docs/general_plan.md`, not in
`docs/05_evaluation_plan.md`, not in `manuscript/paper.md`. It is not that the authors
answered it badly; it is that the question is not asked, and it is the first question
a benchmark whose selling point is "known ground truth" has to answer.

Its consequence is that the paper's central empirical claim — *"when a CF method claims
a change would produce an outcome, does the world agree? … the answer, on every method
wired here, is either no or only vacuously yes"* — has a competing explanation the paper
cannot currently rule out: **in this benchmark, at these settings, nothing any method
could propose would produce an outcome, because the causal pathway from intervention to
label carries almost no signal.** `C_world_oracle ≈ 0.007–0.019` in every single-slice
row of both `table_pns_do_complexity_*` files, and the paper's own §7 records the
unintervened world-label rate as 0.00, is the same fact stated from the other side. The
methods may well be as bad as claimed. This evidence cannot distinguish that from a
generator with no causal effect to find.

---

## Threats to validity the authors have not acknowledged

Beyond the numbered concerns above:

**T1 — Circularity of implementation between generation, oracle, and control.**
`CFfaith.score` builds its reference trajectory by calling
`mechanism.forward_numpy(...)`. `NlinearSCMT.generate` builds the data with the same
call. `CARLARecourse` builds its counterfactual with the same call. So "CARLA scores
`rollout_hard = 1.00`" is a float-equality check of one function against itself, and any
bug in `forward_numpy` would be invisible: it would appear identically in the data, in
the oracle, and in the control, and every consistency check in the suite would still
pass. `CLAUDE.md` records that the second implementation of the ground truth (the
`causal_tscf_bench` cluster) was deleted 2026-08-03 as "a second, untested implementation
of the benchmark's ground truth". That deletion was defensible on maintenance grounds
and it removed the only artifact capable of differentially testing the truth this
benchmark sells. A benchmark whose value proposition is "known truth" should carry an
independent, deliberately-differently-written reference implementation of the
mechanism and the oracle, and a test that they agree. Golden-file tests
(`test_golden_linear.py`) pin the implementation to itself, not to the mathematics.

**T2 — The benchmark grades itself, and Axis A is where it shows.** `general_plan.md`
§5 states Axis A is "near-inert on the synthetic tiers, and that is itself a finding".
Axis A scores the *benchmark*; its inertness is a property the designers chose (FATAL-1),
reported as a discovery. The manuscript then uses that same inertness (H2: "explainers'
failures are propagation failures, not graph-estimation failures") to strengthen the
claim about methods. That is a closed loop: the generator is parameterised so the graph
barely matters, the metric confirms the graph barely matters, and the confirmation is
cited as evidence that the methods' failures are elsewhere. `general_plan.md` §6 already
contains the correct principle ("increasing realism buys generality, and buys nothing at
all about construct validity") — it needs a companion: *a benchmark's own diagnostics
cannot validate the benchmark's own design choices.*

**T3 — `Δ_outcome` is a classifier property inside a method-ranking statistic.**
`Δ_outcome = B − C` measures LSTM-vs-SCM-label-rule disagreement. It is summed into
`Δ_total`, the declared ranking axis, and in `schedule` mode it *is* `Δ_total`
(FATAL-2). One LSTM per config is trained, and the headline tables do not carry its test
accuracy. The paper knows this matters — it documents `full_interior_label` becoming
unusable at 0.501 accuracy, and correctly reasons about the 0.79-vs-0.92 gap in H4
evidence (ii) — but does not carry the lesson into the ranking axis. A method with
larger edits will push trajectories further from the LSTM's training distribution and
score a larger `Δ_outcome` with no causal content whatever.

**T4 — Method-count double-dipping against the project's own rank-statistic rule.**
`05_evaluation_plan.md` §6 requires "Methods for any rank statistic: ≥ 6 **and a
non-degenerate column**". The six methods in every flagship table include **two positive
controls whose scores are construction-true** (CARLA, PearlCARLA — the paper says so
itself) and exclude `CftsCounts` (known to hang) and `TSCausalCF` (the primary
third-party foil, wired into the default recipe 2026-08-06 but never run at
`full`/`full_nl`). The independent evidence base is **four** third-party methods, two of
which sit at validity ≈ 0.5. The paper's own rule is not met, and the most important
omission is the one method whose failure would be a claim about the literature rather
than about a vendored repo — `TSCausalCF` is Bahri et al., the declared Related Work
foil, and §3's entire "penalise vs derive" argument is currently unsupported by any
measurement of the thing being critiqued.

**T5 — Non-determinism in an evaluated method, not resolved.**
`05_evaluation_plan.md`'s M4b reproducibility note records `CftsConfeti`'s no-op count
varying 12/27 → 10/27 across two runs of an unchanged tree — "some non-deterministic
component in that method's search (not seeded through by this pipeline)". The same
method has the lowest scorable-`n` in every synthetic table (79/150, 74/150; MAJOR-9).
The paper flags it and does not chase it. If `CftsConfeti`'s output is not
seed-reproducible, none of its rows are, and the seed-cluster bootstrap's clustering
assumption is violated for that method.

**T6 — The horizon result is confounded with classifier readout, and the paper found it
but under-generalised it.** `05_evaluation_plan.md` H4 evidence (ii) records that LSTM
accuracy tracks the carry distance from label site to terminal readout
(0 → 0.996, 9 → 0.942, 11 → 0.790, 39 → 0.501), and concludes "the same contraction that
stops an intervention reaching a distant label stops a classifier reading one". This is
an excellent observation, and it means **validity-vs-horizon curves confound the SCM's
contraction with the LSTM's memory decay** on *every* config, not just the interior-label
one. Contribution 2's "reporting rule" then rests on a quantity with two mechanisms and
no way to separate them, because the classifier's readout distance and the intervention's
propagation distance are the same number under terminal labelling. The fix (a
non-recurrent or attention-based readout as a control) is cheap and would materially
strengthen the one contribution I think survives.

**T7 — `n_cf` inconsistency across seeds within one config.** `full_nl` seeds 1–2 ran at
`n_cf = 10` against seed 0's `n_cf = 20` after runtime failures. The paper states this
plainly, which I credit. But a cluster bootstrap over 3 clusters of unequal size, two of
which are half the size of the third, has an effective sample smaller than the nominal
one, and the resulting CIs are reported alongside `full`'s as if comparable.

---

## What I would accept

The evidence in this repository, as it currently stands, legitimately supports the
following and nothing broader:

> **We construct a temporal additive-noise SCM benchmark with exact structural
> counterfactuals, and use it to define and validate three instruments for auditing
> counterfactual explanations against a known mechanism: (i) do-complexity `D`, the
> number of timesteps a proposal must declare as interventions before the mechanism can
> reproduce it — a reading-invariant, per-method quantity on which six wired methods
> span 1 to 100 out of a 100-step trajectory; (ii) a vacuity detector showing that a
> proposal can satisfy every model-side CF metric while intervening on nothing; and
> (iii) a demonstration that recourse validity is not a property of the method alone but
> of the intervention-to-label-site distance, with the direction of that dependence
> reversing between contracting (MLP/VAR) and non-contracting (spring/Kuramoto)
> mechanism families — so no temporal-recourse validity number is interpretable without
> its stated `t_label − t0` and the mechanism's measured contraction rate. All results
> are established at 3 seeds on synthetic data whose causal graph carries a measured
> fraction of the generating signal, which we report per configuration; the audit's
> `Δ` decomposition and the CF-faith gate are reported as diagnostics whose dynamic
> range is bounded by that fraction, not as method rankings.**

That is a solid workshop paper or a strong short conference paper. `D` is a genuine,
novel, reading-invariant contribution and the "methods buy validity by intervening
almost everywhere" observation is the paper's best sentence — it survives FATAL-1
because `D` is computed from the *proposal*, not from the mechanism's response, and so
is immune to the coupling-strength problem that hollows out everything else. The
vacuity finding (`frac_vacuous = 1.00` beside CARLA's `rollout_hard = 1.00`) is likewise
real and genuinely useful to the field. The horizon reversal on spring/Kuramoto —
12/12 comparisons, unanimous direction, opposite to the dissipative case — is the
cleanest designed falsification test in the document and it landed.

What it cannot currently support: "a benchmark and audit protocol … evaluated on
whether their claimed causal efficacy is realised by the data-generating process", as a
claim about CF methods. On the configurations run, the data-generating process realises
almost no causal efficacy for anyone, and the paper cannot yet separate "methods fail"
from "the generator has nothing to find".

---

## Minor issues

1. `general_plan.md` §7 misstates the committed table on two counts (PearlCARLA `NaN`
   vs `0.0`; "every other method exactly 0.00" vs CftsCels `pearl_hard = 0.192`). R5
   requires the doc be fixed in the same commit as the disagreement is found.
2. `05_evaluation_plan.md` §1 lists `smoke_spring` as "5 particles (20 feat.)"; the
   preset is `k=10`. `smoke_kuramoto` is `k=5`. Also both list `T` as "TBD" while runs
   used `T=30`.
3. `MLPMechanism`'s docstring asserts the input scaling puts pre-activations "in tanh's
   curved region (genuine nonlinearity)". Measured mean `|z| = 0.015`. R5 applies.
4. `SMOKE_NONMONOTONIC`'s docstring says the ablation varies "the hidden
   representation's monotonicity"; it varies it by 2.3e-8 of data variance. Same fix as
   FATAL-1/3.
5. `table_graph_quality_*.csv` repeats `propagation_error_min/max` identically on every
   row of a config — they are config-level constants pooled into a per-row column, which
   reads as a per-corruption-level statistic. Move to a caption or a separate table.
6. The `soft` CF-faith `scale` parameter is never justified or sensitivity-tested, yet
   `fig1` plots it as the y-axis. State it and sweep it.
7. `docs/axis_metrics_report.md` line 63–67 mark CF-faith R6-green; per MAJOR-4 the
   status should be qualified ("indicator validated at the poles; no intermediate
   discrimination test") rather than a bare ✅.
8. `manuscript/paper.md`'s abstract says "declaring 6 to 100 of a 100-step trajectory as
   interventions" while §1 says "83 to 100" — the discrepancy is CftsCels (`D` = 6.6 on
   `full`). Pick one and use it consistently.
9. `smoke_spring`'s `axis_a` reports `LagAcc = 0.0` and `LagF1 = 0.0` alongside a
   non-trivial `AUC`; worth a sanity check that the lag-indexing convention matches for
   the physical families.
10. The paper repeatedly cites CausalTimePrior (arXiv 2603.11090) and CAUKER as
    independent corroboration. `docs/risk_register.md` RISK-06 already flags the novelty
    contest; I'd route these to the Literature agent for verification before submission,
    since the "independent confirmation that the counterfactual gap is open" argument is
    load-bearing for §3 and rests on a quotation from an appendix.

---

## Handoffs

| Finding | Owner | Blocking? |
|---|---|---|
| FATAL-1 coupling share; re-parameterise generator, add coupling-share reporting | Benchmark Generation / Experimental | **Yes** |
| FATAL-2 `Δ_total` degeneracy; intermediate reading or demote to diagnostic | Methodology & Theory | **Yes** |
| FATAL-3 H5 evidence (ii) withdrawal + ablation precondition gate | Experimental / Statistical | **Yes** |
| MAJOR-4 CF-faith intermediate R6 tests; `hard` tolerance sweep; §7 correction | Metric & Axis / Results & Writing | **Yes** |
| MAJOR-5 negative (no-op) control; PearlCARLA `full_nl` resolution | Experimental | Yes |
| MAJOR-6 discovery degeneracy assertions; spring below-chance AUC bug hunt | Experimental | Yes |
| MAJOR-7 matched-scale graph-quality comparison; per-seed curves | Statistical | Yes |
| MAJOR-8 seeds ≥ 20 or no CIs | Experimental / Statistical | **Yes** |
| MAJOR-9 attrition-robust CF-faith reporting; `INTERVENTION_TOL` sensitivity | Metric & Axis | Yes |
| FIXABLE-10 re-stamp sweeps; regenerability CI check | Software Quality | No |
| T1 independent oracle reimplementation | Methodology & Theory / Experimental | Yes |
| T4 run `TSCausalCF` at `full`/`full_nl` | Experimental | **Yes** — §3's central foil is unmeasured |
| T6 non-recurrent readout control for the horizon claim | Experimental | No (strengthening) |
| Minor 10 novelty/citation verification | Literature | No |
