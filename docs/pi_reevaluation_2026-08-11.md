# PI Re-evaluation — CausalTemp-XAI

**Date:** 2026-08-11
**Branch:** `contribution-refocus`
**Author:** PI
**Status:** Decision memo. Supersedes no document by itself — every change it
recommends requires its own recorded decision in `DECISIONS.md` / `ROADMAP.md`
per `CLAUDE.md`'s scope rule.

---

## 0. Summary of the verdict

Two of the six reported problems are **measurement-invalidating artifacts** that
must be fixed before anything downstream of them is reported (Findings 1 and 5).
Two are **valid measurements being reported in an invalid frame** (Findings 3 and
4). One is **partly confirmed with the reporter's own numbers wrong in one cell**,
and the underlying defect is worse than reported (Finding 2). One is **confirmed
outright against the project's own pre-registration** (Finding 6).

The headline consequence:

> **`NlinearSCM-T` is not a nonlinear SCM.** On `full_nl` a best-fit affine map
> reproduces the mechanism with `R² = 1.000000` on all ten channels; the
> nonlinear residual is **1.7×10⁻⁴ of the innovation-noise standard deviation**.
> Every claim of the form "the protocol / the finding carries over from linear to
> nonlinear" is currently unsupported, because there is only one tier and it is
> linear. *(High confidence — direct instrumentation, §1.)*

I also found **three defects nobody listed**, one of which flips the sign of a
published column (§7).

The test suite is **568 passed, 0 failed** (`uv run --no-sync python -m pytest
tests/ -q`, this working tree). R1 is satisfied. That is not reassurance — it is
the finding. A green suite coexists with a numerically-linear "nonlinear" tier
and a graph-discovery table reporting the sklearn constant-predictor convention
as a measurement. No test asserts that the nonlinear mechanism is measurably
nonlinear; no test asserts that a discovery score matrix is non-constant.

**Method note.** I did not re-run the experiment pipeline. All new numbers below
come from direct instrumentation of `benchmarks/mechanisms.py` /
`benchmarks/generator.py` and one smoke-scale DYNOTEARS refit, plus reading of
the committed `results/` artifacts. Scripts are in the session scratchpad and are
trivially reconstructible from the descriptions given.

---

## 1. Finding 1 — the nonlinear tier is linear

### Verdict: **CONFIRMED, and materially worse than reported. This is a bug/artifact, not a negative result.**

The reported empirical observation (the two datasets are numerically
indistinguishable) reproduces, and I confirmed the downstream signature
independently: `results/tables/table_seed_aggregate_smoke_nl_lstm.csv` and
`..._smoke_nonmonotonic_lstm.csv` agree to 3–4 significant figures on **every**
method (CARLA `proximity_l1` 12.588132 vs 12.586030; CftsCels 0.5085600 vs
0.5084940; identical validity on 6 of 8 methods).

But the diagnosis is not "the ablation is weak". It is that **the mechanism the
ablation perturbs is operating in its linear regime, and so is the entire tier.**

**Pre-activation instrumentation** (empirical windows the generator actually
visits, 20 000 sampled windows):

| | `smoke_nl` | `full_nl` |
|---|---|---|
| state std | 0.1718 | 0.1761 |
| hidden pre-activation `\|z\|` mean | 0.0137 | 0.0152 |
| `\|z\|` 99th percentile | 0.0936 | 0.0822 |
| `\|z\|` max | 0.597 | 0.332 |
| fraction of `\|z\| > 1.0` | **0.0** | **0.0** |
| mean `\|tanh(z) − z\|` | 1.51e-05 | 1.04e-05 |
| output-branch pre-activation `\|out\|` mean | 0.0094 | 0.0096 |

The `1/√(n_active_parents)` input scaling documented at `mechanisms.py:236-240`
as making pre-activations "land in tanh's curved region (genuine nonlinearity)"
does the opposite: it is a *shrinkage*. Combined with `spectral_cap = 0.9` and a
state distribution of std ≈ 0.17, pre-activations never leave the interval where
`tanh(z) = z` to five decimal places.

**Nonlinearity quantification** — R² of the best affine map from the lag window to
the mechanism's output, over the empirical window distribution:

| | `smoke_nl` | `full_nl` |
|---|---|---|
| R² (min over channels) | 0.999999 | **1.000000** |
| nonlinear share of output variance | 2.9e-07 | **6.7e-08** |
| RMS nonlinear residual | 4.16e-05 | 2.44e-05 |
| innovation-noise std (Laplace, scale 0.1) | 0.1414 | 0.1414 |
| **nonlinear residual ÷ noise std** | 2.9e-04 | **1.7e-04** |

The nonlinear component of the "nonlinear" SCM is roughly **5 800× below the
noise floor of the same SCM**. It is not detectable in the generated data by any
estimator, at any sample size, because it is dominated by the innovation noise
the generator adds on top of it.

**Branch decomposition** (this is the design fault, and it is two independent
failures compounding). `mean_i = decay_i·x_{t−1}^i + gain·tanh(MLP_i(masked
lags))`. The first term is *exactly* linear and is the only term with no graph
dependence:

| `full_nl` | value |
|---|---|
| variance of the decay branch | 1.089e-02 |
| variance of the MLP branch | 1.141e-04 |
| **MLP branch share of output variance** | **1.03 %** |
| R² of a linear fit to the **MLP branch alone** | 0.9999965 |
| nonlinear share *within* the MLP branch | 3.5e-06 |

So: 99 % of the mechanism is a scalar per-channel decay, and the remaining 1 % is
itself linear to six decimal places. `smoke_nl` is the same picture (MLP branch
1.58 % of variance, R² 0.9999932).

**This is not fixable by raising the noise scale alone.** I re-evaluated the same
mechanism on inputs scaled ×10 and ×50. At ×50 the nonlinear share *within the
MLP branch* rises to 0.14 — but the MLP branch's share of total variance *falls*
to 0.24 %, because the exactly-linear decay branch scales linearly while the
`gain·tanh(·)` branch saturates at `gain = 0.8`. Total-mechanism R² stays at
0.9982. The fix requires changing `decay_range`, `gain`, `spectral_cap` and
`init_gain` together, or replacing the additive-decay parameterisation — a
mechanism redesign, not a hyperparameter nudge.

**Consequences.**

- **H5 evidence (ii) (non-monotonicity) is vacuous and must be retracted, not
  restated.** Swapping `tanh`→`sin` changes the one-step output by mean
  3.55e-06, max 8.29e-04 on `full_nl` — 3.4e-05 of the output RMS, and three
  orders of magnitude below the innovation noise. The ablation perturbs nothing.
  Reporting it as a refuted prediction implies it was tested. It was not.
- **H5 as a whole is weakened.** It is a synthesis over three refutations; one leg
  is now known to be a null perturbation, so it rests on two.
- **`CLAUDE.md` Architecture, `general_plan.md` §4.3, §6 (Tier 1), and every
  "carries over unchanged from linear to nonlinear" claim are unsupported.** The
  *mathematical* claim (additive noise ⇒ exact abduction ⇒ CF-faith and the
  oracle CF transfer) remains true as a theorem. The *empirical* claim — that the
  benchmark demonstrates this on a nonlinear mechanism — does not hold, because
  the second mechanism family is the first one with extra steps.
- **The two-mechanism-family axis is the project's stated generalisation
  evidence** (`general_plan.md` §8: "the two-mechanism-family axis (linear VAR vs
  nonlinear MLP) is the stated generalisation evidence", offered in place of the
  descoped classifier axis). That evidence does not currently exist. This is the
  single most damaging item in this memo.

*(Confidence: High. Direct instrumentation of the shipped mechanism on the
shipped configs; three independent quantifications agreeing.)*

---

## 2. Finding 2 — CF-faith `hard`

### Verdict: **PARTLY CONFIRMED. The reported numbers are wrong in one cell; the actual defect is structural and deeper than "tautological".**

**Correction to the evidence as given.** The claim "CftsWachter / CftsCOMTE /
CftsCels / CftsConfeti: rollout_hard = 0.0, pearl_hard = 0.0 (all of them, both
configs)" is **false**. On `full`, `CftsCels` scores
`cf_faith_pearl_hard = 0.1919 [0.1143, 0.2762]`, n = 99, and it is stable across
seeds (0.171 / 0.229 / 0.172). One non-control method does clear the gate on one
semantics roughly one time in five.

**`docs/general_plan.md` §7 is stale and contradicted by the committed table** —
an R5 violation. §7 states: *"On `full`, post-`d3c377d`, the column reads CARLA
1.00, PearlCARLA NaN, and every other method exactly 0.00"*. The table says
PearlCARLA `rollout_hard = 0.0` with n = 141 (not NaN), and CftsCels
`pearl_hard = 0.192` (not 0.00). Two of three assertions are wrong.

**The structural defect.** `hard` and `soft` are **not two metrics**. From
`metrics/cf_faith.py:241-244`:

```
l1_residual = mean |x_cf[t0:] − reference[t0:]|
hard = float(l1_residual < 1e-3)
soft = float(exp(−l1_residual / 1.0))
```

They are one statistic — the mean post-intervention mechanism residual `r` — read
at two thresholds. So:

1. **`hard` is a construction detector, not a quality metric.** Its tolerance
   `1e-3` is **1/176 of `full_nl`'s state standard deviation (0.176)**. It asks
   "is this CF the exact mechanism rollout to four decimal places?" Only a method
   that *emits* that rollout can pass. That is why exactly the two positive
   controls pass, each under exactly its own semantics. Calling this "a benchmark
   result" (`CLAUDE.md`, `general_plan.md` §7) is defensible only in the narrow
   sense that it demonstrates the two semantics are mutually exclusive — which is
   a property of the *definitions*, provable on paper, not an empirical finding
   requiring six methods and three seeds.
2. **`soft` is not normalised and its `[0,1]` range is misleading.** `scale = 1.0`
   with `r` in **raw state units**. Converting the observed `full_nl` range back
   to residuals: `soft = 0.823` (CftsCOMTE) ⇒ `r = 0.195` ⇒ **1.1 state σ**;
   `soft = 0.923` (CftsCels) ⇒ `r = 0.081` ⇒ 0.46 σ; `soft = 0.881` (CftsWachter)
   ⇒ 0.72 σ. A reader seeing "0.88" reads a high score; the actual content is
   "this CF deviates from the mechanism by three quarters of the process's own
   standard deviation at every post-intervention step". The apparent dynamic
   range 0.67–1.00 is an uncalibrated Laplace kernel, not a calibrated score.

**Where the headline should rest.** Not on `hard`, and not on `soft` as printed.
It should rest on **the mean post-intervention mechanism residual, reported in
units of the process's own standard deviation**, with `hard` demoted explicitly
to "identifies which construction, if any, the CF is". That change is a
presentation fix — the underlying numbers are already computed and already have
green adversarial tests (`docs/axis_metrics_report.md` rows for CF-faith rollout/
pearl hard & soft are all ✅), so **R6 is satisfied and this is not a re-run
item.**

*(Confidence: High on the arithmetic and the code reading; Medium on the
recommendation that σ-normalisation is the right presentation — an alternative is
to normalise by the innovation scale, and the choice must be stated either way,
see §3.)*

---

## 3. Finding 3 — graph quality barely moves the outcome

### Verdict: **PARTLY CONFIRMED. My own leading hypothesis was REFUTED by the measurement and I am recording that against myself. But a separate, decisive defect stands: the cross-family "dynamic range" is a units artifact.**

The table reproduces as reported
(`results/tables/table_graph_quality_full_nl_vs_smoke_spring_vs_smoke_kuramoto.csv`):
`full_nl` 0 → 0.00325 [0.00183, 0.00401] over SHD 0 → 35.3; `smoke_kuramoto`
0 → 0.1457; `smoke_spring` 0 → 0.3990.

**Hypothesis I tested (and had to abandon).** Given Finding 1 — the graph enters
`MLPMechanism` *only* through the parent mask on the MLP branch, which carries
1.03 % of output variance — the obvious explanation is that corrupting the graph
on `full_nl` can only perturb 1 % of the mechanism, whereas in Spring and
Kuramoto the graph *is* the coupling and therefore the whole mechanism. If true,
`general_plan.md` §3's attribution of the contrast to **dissipation** would be
confounded and unidentified.

I measured it directly: same mechanism, same visited windows, graph fully rewired
(edge count preserved), one-step output shift:

| family | mean abs one-step output shift under full corruption | as fraction of the mechanism's own one-step change |
|---|---|---|
| `full_nl` (mlp) | 0.0119 | 0.201 |
| `smoke_kuramoto` | 0.0219 | 0.175 |
| `smoke_spring` | 0.0593 | 0.514 |

**The hypothesis fails.** Per-step graph sensitivity differs by only 2–5× across
families, not the 45–123× the reported `graph_error` range implies. Dividing the
observed `graph_error` residual by the measured per-step sensitivity gives an
accumulation factor of **0.27× for `full_nl` against 6.7× for both
non-dissipative families** — a ~25× difference in *accumulation*, which is
exactly what dissipation predicts. `general_plan.md` §3's dissipation attribution
survives this test. I am recording this because the hypothesis was mine and the
evidence went against it.

**What does not survive: the cross-family comparison as published.**
`graph_error = cf_faith_gt − cf_faith_inferred` (`metrics/axis_a.py:169-194`),
and `cf_faith` here is the **soft** score, i.e. `exp(−r)` with `r` in **raw state
units** and `scale = 1.0` (§2). The three families' state scales differ by 66×:

| family | state std | `graph_error` (corrupt=1) | implied residual `r` | `r` in units of the family's own σ |
|---|---|---|---|---|
| `full_nl` | 0.176 | 0.00325 | 0.00325 | **0.0185 σ** |
| `smoke_kuramoto` | 11.674 | 0.1457 | 0.1574 | **0.0135 σ** |
| `smoke_spring` | 6.934 | 0.3990 | 0.5083 | **0.0733 σ** |

**Normalised by each process's own scale, the claimed 45×–123× dynamic range
collapses to ≤ 4×, and the Kuramoto / `full_nl` ordering reverses.** Under a
different defensible normaliser (the mechanism's own one-step change) the range
survives at 23–80×. The published number uses **neither** normaliser and states
neither. A quantity whose headline ratio moves between "4× and inverted" and
"80×" depending on an unstated normalisation choice is not reportable as
evidence.

(Kuramoto's σ is itself dominated by deterministic phase drift `ω·t` on an
unwrapped phase, which is why I give both normalisers rather than picking one.
That ambiguity is real and must be resolved explicitly, not silently.)

**Additionally**, the `dynotears` rows of that same table for `smoke_spring`
(0.269) and `smoke_kuramoto` (0.126) are computed against a **degenerate empty
inferred graph** — see Finding 5. The Kuramoto `dynotears` row is arithmetically
the `corrupt_frac = 1` row wearing a different label; it is not a discovery
result and must not be printed as one.

**Net.** The *phenomenon* (graph quality matters where effects persist, not where
they dissipate) is probably real and is honestly caveated in `general_plan.md`
§3. The *quantification* is not currently defensible, and H2's "graph error
≤ 0.004 even for a random graph at full scale" is a number measured on a
mechanism whose graph-gated branch carries 1 % of its variance — it must be
re-measured on a genuinely nonlinear tier before it is cited.

*(Confidence: High that the units problem invalidates the cross-family ratio as
stated. Medium that the dissipation attribution is correct — it survived one
direct test, which is not the same as being established.)*

---

## 4. Finding 4 — PearlCARLA degenerates on `full_nl`

### Verdict: **CONFIRMED as an invalid measurement. Diagnosed as an optimisation collapse that is entailed by the project's own horizon result — it is not an independent scientific finding about Pearl semantics, and it must not be reported as a method row.**

All three seeds are identical in kind
(`results/full_nl_seed{0,1,2}/lstm/eval_PearlCARLA.json`): `validity = 0.0`,
`frac_vacuous = 1.0`, `frac_degenerate = 1.0`, `n_cf_faith_scorable = 0`,
`do_complexity_mean = 0.0`, `frac_altered = 0.0`, `proximity_l1` = 4.3e-06 /
1.1e-05 / 2.7e-05. The method returns a numerically zero edit.

**Diagnosis (High confidence).** `experiments/03_run_cf_methods.py:102` runs
PearlCARLA at `t0_fractions=(0.25, 0.5)` ⇒ `t0 ∈ {25, 50}` at `T = 100` ⇒
**`T − t0 ∈ {50, 75}`**. The project's own measured collapse horizon for
PearlCARLA on `full_nl` is **8.25 [7.14, 12.86]**
(`results/tables/table_collapse_horizon_full_vs_full_nl.csv`,
`frac_boot_crossed = 1.0`). **The method is being run 6–9× beyond its own
measured feasibility horizon.**

At feasible horizons on the same config it works: `T − t0 = 5` ⇒ validity 0.933;
`T − t0 = 10` ⇒ 0.267; `T − t0 = 20` ⇒ 0.0
(`table_horizon_full_vs_full_nl.csv`).

The mechanism of the collapse is understood and already recorded. The Pearl delta
obeys the homogeneous recursion; I measured `full_nl`'s empirical one-step
Jacobian gain at **0.55**, so `0.55^50 ≈ 6e-14` — below float32 resolution. The
prediction-loss gradient w.r.t. `delta` vanishes, the `lam_prox` term is the only
surviving gradient, and Adam drives `delta → 0`.
`docs/archive/m2_multiseed_and_pearl_carla.md` §2.3's proposed remedy (lower
`lam_prox`) was **explicitly ruled out on 2026-07-30** for exactly this reason.

**Therefore it is both, and the framing matters.** It is a genuine property of
Pearl-semantics recourse under contraction (which is H4, already the project's
own claim) *and* a numerical failure of this particular Adam loop. Those two are
**not separable from this run**, because at `T − t0 = 50` there is no gradient to
tell them apart. Printing `PearlCARLA validity 0.0, proximity 1.4e-05` in the same
table as `CftsWachter validity 0.98` asserts a comparison that was not made: one
row measures a method, the other measures the configuration.

**Required action:** withdraw the `full_nl` PearlCARLA row from every
method-comparison table and replace it with an explicit "no feasible recourse at
`T − t0 = 50/75`; measured collapse horizon 8.25 [7.14, 12.86]" annotation. Then
re-run the roster at a `t0` inside the feasible horizon if a comparison is
wanted. The `full` (linear) PearlCARLA rows are also at `T − t0 ∈ {50, 75}`
against a collapse horizon of 28.9 [26.7, 32.0] — same problem, less extreme, and
they show it (`frac_vacuous` 0.00 / 0.02 / 0.16, validity 0.02).

**Related, and not in the reported evidence:** CARLA on `full_nl` at `T − t0 = 40`
has `frac_vacuous = 1.0`. The pooled "CARLA validity 0.667 [0.0, 1.0]" on
`full_nl` is partly a vacuous-intervention number. RISK-17 covers this; the
tables do not yet reflect it.

---

## 5. Finding 5 — degenerate graph-discovery numbers

### Verdict: **CONFIRMED for Kuramoto (degenerate; the AUC is not a measurement). REFUTED for Spring (real measurement against a partial ground truth). The specific "SHD exactly 8.0" inference is a pooling artifact and does not support the conclusion it was offered for — though the conclusion is right for Kuramoto by a different route.**

**Kuramoto — degenerate, and I traced the exact cause.** I refit DYNOTEARS on
`smoke_kuramoto` (k=5, T=30, N=500, seed 0):

| | value |
|---|---|
| true edges | 5 |
| inferred edges at threshold 0.1 | **0** |
| unique values in the score matrix | **1** |
| score matrix max | **0.0** |
| raw `a_est` non-zeros | 5 |
| raw `a_est` max abs | **0.9877** |

DYNOTEARS *did* recover five strong lag-1 coefficients — and every one of them is
a **self-loop**. Kuramoto's update is `θ_t = θ_{t−1} + dt·(ω + coupling)`, so the
lag-1 self-coefficient is ≈ 1.0 by construction (0.9877 measured), while the
cross-coupling term is `dt·K/deg·sin(Δθ) ≈ 0.5×0.1/deg ≈ 0.025` — a **40:1**
ratio that `lambda_a = 0.01` L1 shrinkage removes entirely.
`DYNOTEARS.inferred_graph` then zeroes the diagonal
(`dynotears.py:207-210`), leaving a score matrix that is **identically zero**, and
`smax = 0` skips the normalisation.

`metrics/axis_a.py::graph_auc` guards only a degenerate `y_true`; it does **not**
guard a constant `y_score`. `roc_auc_score` on a constant score vector returns
exactly 0.5 by convention. **So the reported "AUC 0.500 (± 0.000, 3 seeds)" is
the sklearn constant-predictor convention, not a measurement of discovery
performance.** `shd_between_methods = 0.0` on all three seeds follows trivially
(both methods return the empty graph), as does `LagAcc = 0.0` and `LagF1 = 0.0`
in every Kuramoto `graph_error.json`. Reporting it as "AUC" is **not
defensible.**

**Spring — not degenerate; the reported reading is wrong.** Same refit:
score matrix has 12 distinct values, 7 inferred edges, `a_est` nnz = 21. DYNOTEARS
AUC 0.460 [0.458, 0.464] is a real, systematically **below-chance** measurement.
The cause is construct validity, not degeneracy: `SpringMechanism`'s own
docstring states that each particle's position←own-velocity relation and its own
position term in its own acceleration are **deliberately excluded** from `graph`
as "baseline/structural". A discovery method that correctly recovers those — and
they are the *strongest* dependencies in the system — is scored as a false
positive against a ground truth that omits them by convention. AUC-vs-"true
graph" on Spring measures agreement with a **partial** ground truth. Below-chance
is the expected result of that convention, not a method failure.

**The "SHD exactly 8.0" inference is wrong.** Per-seed
`dynotears_shd_to_true` on Kuramoto is 10 / 4 / 10 and on Spring 10 / 4 / 10
(`results/tier1_suite/summary.json`). 8.0 is the *mean* of a varying quantity, not
a constant. The degeneracy conclusion for Kuramoto is correct; the route offered
to it is not, and would not have survived review.

**Required actions.** (a) `graph_auc` must return `NaN` when the score matrix is
constant, with a test. (b) The Kuramoto rows of
`results/tier1_suite/table_method_suitability.csv` and the Kuramoto `dynotears`
row of the graph-quality table must be withdrawn. (c) The Spring rows may stand
**only** with an explicit statement that the reference graph omits the
position←velocity and self-position structure by design, which is why
below-chance AUC is expected — or the reference graph must be extended and the
row re-measured.

*(Confidence: High for Kuramoto — I reproduced the empty score matrix and the
self-loop cause directly. High for Spring's non-degeneracy. Medium for the
"partial ground truth explains below-chance" attribution — it is the documented
convention and it predicts the sign, but I did not run the counterfactual with an
extended reference graph.)*

---

## 6. Finding 6 — statistical power

### Verdict: **CONFIRMED. The pre-registered evidence standard is not met by any headline, and the project's own document says so in places while the claims proceed anyway.**

`docs/05_evaluation_plan.md` RQ1 sets the standard explicitly: **"≥ 5 seeds ×
n_cf ≥ 100 × LSTM on `full`, bootstrap 95 % CIs"**, with "Hypothesis confirmation
at the **CI level**, never the point estimate."

What was actually run: **3 seeds × n_cf = 50** for the main tables. The horizon
sweeps are worse — `n_cf = 20` for `full` and `full_nl` seed 0, **`n_cf = 10` for
`full_nl` seeds 1–2** (reduced after runtime failures, per §RQ4). Pooling three
seeds with unequal `n_cf` into one bootstrap CI is not a valid pooling; the seeds
are not exchangeable.

The verified examples are exactly as reported: CARLA validity on `full` 0.213
[0.000, 0.633]; on `full_nl` 0.667 [0.000, 1.000] — an interval spanning the
entire support of the statistic.

**The deeper problem is not "wide CIs".** With 3 seeds, a percentile bootstrap
over seed-level means resamples from a 3-point empirical distribution. The 2.5th
and 97.5th percentiles are effectively the minimum and maximum of the three seed
means. The interval is not a 95 % interval in any useful sense; it is the range,
relabelled. Every "the CI excludes 0" and "the CIs are non-overlapping" statement
in `05_evaluation_plan.md` §5 built on n = 3 is uninterpretable.

**Seed count required.** To separate two methods whose validity differs by 0.15
with an observed per-seed standard deviation around 0.25 at 80 % power and
α = 0.05 needs roughly **n ≈ 22 seeds per arm**. That is the honest number for a
ranking claim.

Recommended targets:

- **5 seeds** — the pre-registered floor. Mandatory for anything called a verdict.
- **10 seeds** — the practical target for the do-complexity and `frac_vacuous`
  results, whose per-seed variance is much smaller (do-complexity per-seed sd is
  ~1–2 on a 0–100 scale) and which are therefore powered at 10.
- **20+ seeds** — required before any validity-difference ranking claim between
  two non-control methods. If that is unaffordable, the ranking claim must be
  dropped, not reported underpowered.

*(Confidence: High.)*

---

## 7. Three defects nobody listed

### 7.1 `sparsity` is inverted relative to its published definition — **the most citable error in the repo**

`docs/general_plan.md` §5: *"**Sparsity** — L0 fraction of altered features."*

`causaltemp_xai/metrics/axis_c.py:119-128`: *"Fraction of features that are
**unchanged** between original and CF. A higher sparsity score (closer to 1) means
**fewer** features were modified."* Confirmed arithmetically in every eval JSON:
`sparsity + frac_altered = 1.0` exactly (`full_seed0` PearlCARLA: 0.80718 +
0.19282).

**Every sparsity number in every table reads backwards against the specification
document.** `CftsWachter` on `full_nl` reports `sparsity = 0.0298`; a reader
following `general_plan.md` §5 concludes "3 % of features altered", when the code
means **97 % of features altered**. This is an R5 violation with a sign flip, and
it is the kind of thing a reviewer finds in ten minutes and never recovers trust
from. Fix the document (R5: the code is correct), or rename the column
`unchanged_fraction`.

### 7.2 The same method is reported twice under two names

`results/smoke_nl_seed0/lstm/eval_CausalFeasibility.json` and
`eval_TSCausal.json` are **byte-identical except for the `method` field**. Both
appear as separate rows in
`results/tables/table_seed_aggregate_smoke_nl_lstm.csv` with values matching to 16
digits (validity 0.8333, `proximity_l1` 93.54252914259665, every CF-faith column).

`general_plan.md` §8 records the 2026-08-06 rename `CausalFeasibilityCF` →
`TSCausalCF`. The old name was not retired from the runner. Any count of "methods
evaluated", any rank correlation, any leaderboard computed over that table
**double-counts one method**. This directly touches R3's concern (a name must
denote one thing) and it silently inflates `n` in exactly the statistic H6 was
demoted for being underpowered on.

### 7.3 The green test suite is part of the problem

568 tests pass. `docs/axis_metrics_report.md` shows ✅ or 🟢 on every CF-faith,
do-complexity, vacuity and degeneracy row — R6 is genuinely satisfied for the
metrics. And yet the flagship nonlinear tier is linear and a discovery table
publishes a constant-predictor artifact as AUC.

The gap is that every adversarial test targets a **metric**, and none targets the
**benchmark's own generative claims**. Two tests are missing and both are cheap:

1. `test_generator.py::test_mlp_mechanism_is_measurably_nonlinear` — assert the
   nonlinear share of the mechanism's output variance on the empirical window
   distribution exceeds some floor (e.g. 5 %), and that the `tanh`→`sin`
   activation swap changes the output by more than the innovation noise scale.
2. `test_axis_a.py::test_graph_auc_rejects_constant_scores` — assert `graph_auc`
   returns `NaN`, not 0.5, for a constant score matrix.

Neither would have passed on the current code. Both should be written **before**
the mechanism redesign, so the redesign has an acceptance criterion.

---

## 8. What survives

Stated narrowly and specifically. Everything here is unaffected by Findings 1–6,
and I would defend each in review.

1. **The instrument itself — the exact structural counterfactual.**
   `benchmarks/structural_cf.py`'s abduction–action–prediction oracle under
   additive noise, and the two-semantics `CFfaith` gate. The mathematics is
   correct, the adversarial tests are green, and its validity does not depend on
   the mechanism being nonlinear. **This is the contribution.** The external
   corroboration in `general_plan.md` §3 (CausalTimePrior, arXiv 2603.11090,
   App. B, independently identifying unit-level temporal counterfactuals as
   unfilled and requiring a shared noise tensor) stands and is strong. *(High)*

2. **Do-complexity `D` — the strongest surviving empirical result.** `D` counts
   how many timesteps a proposal must declare as `do()` actions before the
   mechanism can reproduce it. It is **mechanism-free**, so Finding 1 does not
   touch it at all. The `full` values (`T = 100`) are a real and striking finding:
   PearlCARLA 1.0, CftsCels 6.6, CARLA 69.7, CftsCOMTE 83.3, CftsWachter 96.4,
   CftsConfeti **100.0**. "Methods buy their claimed validity by declaring most or
   all of the trajectory an intervention" is publishable, is not tautological, and
   needs only seeds. *(High on the phenomenon; the specific numbers need 10 seeds
   per §6.)*

3. **The vacuous-intervention detector and `frac_vacuous` (RISK-17).** CARLA on
   `full_nl` scores `cf_faith_rollout_hard = 1.00` while being **100 % vacuous**.
   That is a genuine, mechanism-independent critique both of the noiseless-rollout
   family and of Bahri et al.'s residual-penalty criterion, and it is exactly the
   pathology `general_plan.md` §3's boxed note charges them with. Fully green
   adversarial tests. *(High)*

4. **The `t0 ≥ T−1` degeneracy gate and the `frac_degenerate` reporting
   discipline.** Returning NaN rather than 1.0 when the forward loop is empty
   caught a real inflation bug (`d3c377d` retracted CftsConfeti 0.46→0.00,
   CftsCels 0.60→0.00). A methodological contribution in its own right, and it is
   the reason Finding 4's PearlCARLA row is visible as unscorable rather than
   silently reported as perfect. *(High)*

5. **The horizon / collapse-horizon result on `full` — the linear tier only.**
   `LinearSCM-T` is honestly what it claims to be, so nothing in Finding 1 touches
   it. PearlCARLA's collapse horizon on `full` is 28.9 [26.7, 32.0] with
   `frac_boot_crossed = 1.0`. The **reporting rule** that follows —
   *no temporal-recourse validity number is interpretable without its stated
   `t_label − t0`* — is `general_plan.md` §4.2's contribution and it survives
   intact. It is also, per Finding 4, a rule this project is currently violating
   in its own tables. *(High for the rule; Medium for the specific horizon numbers
   at n = 3.)*

6. **Tier-1 discovery comparison on the `linear` family only** (DYNOTEARS AUC
   1.000 / SHD 0.0; PCMCIplus 0.975 / SHD 0.67). The `mlp` row (0.909 / 0.897)
   is now known to be a measurement on a *nearly-linear* SCM and must **not** be
   cited as evidence that either method handles nonlinearity — which, read
   correctly, is a mildly embarrassing explanation for why DYNOTEARS (a linear
   structural-equation method) does so well there. *(High)*

**The honest scope.** The defensible contribution is **substantially narrower
than `general_plan.md` §2 claims.** §2 promises "a benchmark and audit protocol
… evaluated on whether their claimed causal efficacy is realised by the
data-generating process", with §6 presenting a four-family Tier-1 suite as
external validity. What is actually supported today is: **an audit protocol,
validated on one linear SCM family, whose principal empirical finding is that
existing temporal CF methods do not propose interventions in the ordinary sense —
they rewrite most of the trajectory (`D`) or exploit vacuous edits
(`frac_vacuous`).** The four-family suite is currently one honest family
(`linear`), one family that is linear while claiming not to be (`mlp`), and two
whose discovery numbers are degenerate or measured against a partial reference
graph (`kuramoto`, `spring`).

---

## 9. What does not survive

Must be **fixed**, **re-run**, or **retracted from the claim**.

| # | Item | Disposition |
|---|---|---|
| 1 | `NlinearSCM-T` as a nonlinear tier | **FIX + RE-RUN.** Mechanism redesign, then regenerate every `*_nl` result. |
| 2 | "Carries over unchanged from linear to nonlinear" — `CLAUDE.md` Architecture, `general_plan.md` §4.3, §6, §8 | **RETRACT the empirical form**; keep the mathematical form (additive noise ⇒ exact abduction) as a theorem, clearly labelled as unvalidated empirically until (1) lands. |
| 3 | H5 evidence (ii), non-monotonicity ablation | **RETRACT.** The perturbation is 3 orders of magnitude below the noise floor. It was not tested. |
| 4 | H5 as stated (invariance across three axes) | **REWRITE** to two axes, or defer until (1) lands and the third can actually be run. |
| 5 | H2 ("graph error ≤ 0.004 even for a random graph at full scale") | **RE-RUN** on the fixed tier. Measured on a mechanism whose graph-gated branch is 1 % of output variance. |
| 6 | `graph_error` cross-family "45×–123× dynamic range" (`general_plan.md` §3, §5 box) | **FIX the reporting.** Normaliser-dependent: ≤4× and order-reversed under σ-normalisation, 23–80× under innovation-normalisation. Pick one, state it, re-derive. |
| 7 | Kuramoto rows of `results/tier1_suite/table_method_suitability.csv`; Kuramoto `dynotears` row of the graph-quality table | **RETRACT.** Constant-score artifact, not a measurement. |
| 8 | Spring AUC rows | **KEEP with an explicit caveat**, or extend the reference graph and re-measure. |
| 9 | PearlCARLA `full_nl` row in every method-comparison table | **WITHDRAW** and replace with the infeasibility annotation. |
| 10 | `general_plan.md` §7's CF-faith column | **FIX (R5).** Two of three assertions contradicted by the committed table. |
| 11 | `general_plan.md` §5 sparsity definition | **FIX (R5).** Sign-inverted against the code. |
| 12 | Duplicate `CausalFeasibility` / `TSCausal` rows | **FIX.** Retire the old name in the runner; delete the duplicate eval JSONs. |
| 13 | Every CI-level verdict in `05_evaluation_plan.md` §5 | **RE-RUN at ≥ 5 seeds**, or restate as point estimates with the power limitation named. |
| 14 | CF-faith `hard` as a reported per-method column | **DEMOTE** to a construction-identification diagnostic; publish the σ-normalised residual instead. |

---

## 10. Revised contribution statement

Proposed replacement for `docs/general_plan.md` §2's scope paragraph. This is
what the evidence currently supports — not what the project hopes to support
after remediation.

> **Scope (revised 2026-08-11).**
>
> **An audit protocol for temporal counterfactual explanations, and one
> synthetic benchmark family that supplies the ground truth it needs.**
>
> Given a proposed counterfactual, the protocol extracts the intervention it
> implies, realises that same intervention through the true data-generating
> mechanism by exact abduction–action–prediction, and reports the disagreement
> between what the explanation claims and what the mechanism delivers. Because
> the benchmark's SCM is additive-noise, abduction is an exact subtraction and
> both potential outcomes are computable per instance rather than bounded.
>
> The protocol's three instruments are: **do-complexity `D`** (how many timesteps
> a proposal must declare as `do()` actions before the mechanism can reproduce
> it); **`frac_vacuous`** (whether a proposal encodes any intervention at all, or
> merely deletes the factual noise); and the **model-vs-world decomposition**
> `Δ_total = Δ_trajectory + Δ_outcome`. CF-faith is the admission gate that runs
> before them, not a ranking metric.
>
> **The principal empirical finding is negative and is about the methods, not
> about the benchmark:** across every wired explainer, apparent validity is
> bought either by declaring most or all of the trajectory an intervention
> (`D` = 66–100 of `T` = 100 for four of six methods) or by emitting an edit that
> encodes no intervention at all (`frac_vacuous` = 1.00 for the noiseless-rollout
> control at long horizon). No wired method proposes an intervention in the sense
> the word is normally used, and the single-`do()` positive control is the only
> one at `D = 1`.
>
> **Validity established, and where.** All construct-validity arguments are
> established on **`LinearSCM-T` (VAR(1)) only** — the one shipped family whose
> mechanism is demonstrably what it claims to be. `NlinearSCM-T` is **withdrawn
> as a nonlinear tier pending redesign**: instrumentation on 2026-08-11 showed a
> best-fit affine map reproduces its mechanism at `R² = 1.000000`, with the
> nonlinear residual 1.7×10⁻⁴ of the innovation-noise scale, so it currently
> provides no mechanism-family generalisation evidence. `SpringSCM-T` and
> `KuramotoSCM-T` are retained for the horizon and graph-quality contrasts only;
> their graph-discovery diagnostics are withdrawn (Kuramoto) or caveated
> (Spring).
>
> **What this work does not yet claim:** that the protocol's findings generalise
> across mechanism families; that graph quality has a quantified, scale-invariant
> effect on counterfactual faithfulness; or any ranking between non-control
> methods — the last is underpowered at the seed counts run to date.
>
> The natural follow-up, once the evaluation scope is stable: propose a CF method
> that scores well against it. Nothing in this plan depends on that.

---

## 11. Prioritised remediation plan

### P0 — blocks every publication claim. Nothing else is worth doing first.

| ID | Action | Owner | Seeds / scale | DoD |
|---|---|---|---|---|
| **P0-1** | Write the two missing adversarial tests (§7.3) **first**, so the fix has an acceptance criterion: `mlp_mechanism_is_measurably_nonlinear` and `graph_auc_rejects_constant_scores`. Both must fail on current `main`. | Metric & Axis | — | Both tests red on `d99f414`, green after P0-2/P0-4. |
| **P0-2** | Redesign `MLPMechanism`'s parameterisation so the nonlinear component is measurable: raise `gain` and `init_gain`, lower or remove the `1/√n_parents` shrinkage, retune `decay_range` downward so the exactly-linear branch does not dominate. **Target: nonlinear share of output variance ≥ 5 %, MLP-branch share of output variance ≥ 40 %, and the `tanh`→`sin` swap changing the output by more than the innovation-noise std.** Re-verify contraction (`ρ < 1`) after retuning — the current dissipation is a *separate* design property and must be preserved deliberately, not lost by accident. | Benchmark Generation | — | P0-1 tests green; contraction rate re-measured and recorded. |
| **P0-3** | Regenerate **every** `*_nl` artifact from the fixed mechanism: `smoke_nl`, `full_nl`, `smoke_nonmonotonic`, `smoke_regime`, and the `mlp` rows of `tier1_suite` and the graph-quality sweep. | Experiment Orchestration | **5 seeds minimum** on `full_nl` (pre-registered floor); 3 at smoke. `n_cf = 100` on `full_nl` per the pre-registration. | New `table_seed_aggregate_full_nl_lstm.csv`; `smoke_nl` vs `smoke_nonmonotonic` now differ by more than 4 s.f. on at least one method. |
| **P0-4** | Fix `graph_auc` to return `NaN` on a constant score matrix; withdraw the Kuramoto discovery rows; add the reference-graph caveat to Spring. | Metric & Axis | — | P0-1 test green; `table_method_suitability.csv` has no `AUC = 0.500 ± 0.000` row. |
| **P0-5** | Fix the two R5 violations: `general_plan.md` §5 sparsity definition (sign-inverted) and §7's CF-faith column (contradicted on two cells). Same commit as the code they describe, per R5. | Results & Writing | — | Every §5/§7 number matches the committed table. |
| **P0-6** | Withdraw PearlCARLA's `full_nl` row; add the `T − t0` vs collapse-horizon annotation to every validity table (this is `general_plan.md` §4.2's own rule, currently violated in-house). | Results & Writing | — | No method-comparison table contains a row with `n_cf_faith_scorable = 0`. |
| **P0-7** | Retire the `CausalFeasibility` alias in `experiments/03_run_cf_methods.py`; delete the duplicate `eval_CausalFeasibility.json` outputs. | CF Methods | — | No config produces two byte-identical eval JSONs under different names. |

**Scope-rule note:** P0-2 changes a benchmark family's mechanism and P0-3 changes
the seed count. Per `CLAUDE.md`, both require a recorded PI decision in
`ROADMAP.md`/`DECISIONS.md` **before** code is written. This memo is the
justification, not the decision.

### P1 — required before the paper's claims are stated at CI level

| ID | Action | Seeds |
|---|---|---|
| **P1-1** | Re-run `full` at **≥ 5 seeds × n_cf ≥ 100** — the pre-registered standard, never yet met. Regenerate `05_evaluation_plan.md` §5 verdicts against it. | 5 (10 preferred) |
| **P1-2** | Re-run the do-complexity / `frac_vacuous` tables — the surviving headline (§8.2, §8.3) — at **10 seeds**. Per-seed variance is low here, so 10 is genuinely sufficient and this is the cheapest route to a powered headline. | 10 |
| **P1-3** | Resolve the `graph_error` normalisation (σ vs innovation scale), record the choice in `DECISIONS.md`, and re-derive the cross-family comparison. Re-measure H2 on the fixed nonlinear tier. | 5 |
| **P1-4** | Re-run PearlCARLA and CARLA at `t0` values **inside** each config's measured collapse horizon, so a like-for-like method comparison exists at all. | 5 |
| **P1-5** | Demote CF-faith `hard` to a construction-identification diagnostic; publish the σ-normalised post-intervention mechanism residual as the continuous quantity. Presentation-only — the numbers exist and R6 is already satisfied. | — |
| **P1-6** | Restate H5 over two axes (noise family, regime stationarity), or defer until P0-3 makes the third testable. | — |

### P2 — worth doing, does not block

| ID | Action |
|---|---|
| **P2-1** | Extend Spring's reference graph to include the position←velocity and self-position structure, then re-measure discovery AUC. Removes the "below chance" artifact honestly rather than by caveat. |
| **P2-2** | Tune DYNOTEARS's `lambda_a` per family, or standardise per-channel, so the 40:1 self-persistence : cross-coupling ratio on Kuramoto does not annihilate the signal. Currently one global setting is applied to families whose scales differ by 66×. |
| **P2-3** | Adopt `RELATIVE_INTERVENTION_TOL` (0.005) as the default. `intervention.py:38-57` already documents that the absolute `1e-3` is incoherent across channels whose σ differs by 5.5× *within one config* — and 66× across families. This is a known-bad magic number with the replacement already written and opt-in. |
| **P2-4** | Sensitivity of `D` and `frac_vacuous` to the tolerance choice, reported as a robustness panel. `CftsCels`'s `D` already swings 1.0 → 3.1 → 13.2 across three tolerance decades (per `intervention.py`'s own docstring) — a reviewer will ask, and the answer should not be discovered in review. |
| **P2-5** | Consolidate the three overlapping related-work statements (§10 below). |

---

## 12. Repo-simplification recommendation

**Recommendation only. I have deleted nothing.**

### 12.0 The governance fact that dominates this section

Almost the entire `docs/` tree is **git-excluded via `.git/info/exclude`**, not
`.gitignore` — including `general_plan.md`, `05_evaluation_plan.md`,
`ROADMAP.md`, `DECISIONS.md`, `PROGRESS_DASHBOARD.md`, `CLAUDE.md`,
`risk_register.md`, `docs/archive/`, and `docs/reviews/`.

**Deleting any of those is irreversible.** There is no git history to restore
from. The only tracked markdown in the repo is:

- `README.md`
- `CONTRIBUTING.md`
- `results/README.md`
- `docs/BenchmarkingTSCFEs.md`
- `docs/pns_metric_design.md`
- `docs/references_verified.md`

Every deletion recommendation below is annotated `[TRACKED — recoverable]` or
`[UNTRACKED — IRREVERSIBLE]`. **Nothing in the second category should be deleted
without an explicit, separate confirmation, and several of them must not be
deleted at all.**

### 12.1 Safe to DELETE

| Path | Status | Why |
|---|---|---|
| `docs/BenchmarkingTSCFEs.md` | `[TRACKED — recoverable]` | 239-line prose literature essay with bracketed numeric citations and no resolvable bibliography in-file. Not in `CLAUDE.md`'s authoritative doc list. Its content is superseded by `docs/04_literature_gaps.md` + `general_plan.md` §3, which are maintained and which `general_plan.md` designates as authoritative. Recoverable from git if wanted back. **This is the single clearest delete in the repo.** |
| `results/*/lstm/eval_CausalFeasibility.json` (all configs) | regenerable output | Byte-identical to `eval_TSCausal.json` except the `method` field (§7.2). `CLAUDE.md`: "`results/` is fully regenerated output; nothing there is source of truth." Delete together with P0-7. |
| `results/full_nl*`, `results/smoke_nl*`, `results/smoke_nonmonotonic*`, `results/smoke_regime*`, and the `mlp`/`kuramoto` rows of `results/tier1_suite/` | regenerable output | Outputs of an invalid mechanism (Finding 1) and a degenerate discovery path (Finding 5). **Do not delete yet** — they are the evidence for this memo. Mark them RETRACTED in `results/README.md` now, delete after P0-3 regenerates them. |

### 12.2 CONSOLIDATE — do not blind-delete

| Paths | Status | Problem |
|---|---|---|
| `docs/03_mathematical_framework.md` (151 lines) **and** `docs/metrics_formal.md` (209 lines) | both `[UNTRACKED — IRREVERSIBLE]` | **Two documents each claiming to be the source of truth for the mathematics.** `03_mathematical_framework.md` opens: "This document is the single source of truth for the mathematical definitions used in the benchmark." `metrics_formal.md` opens: "Drop-in replacement for the prose definitions in draft v0.1. Supersedes the single-timepoint intervention convention." They cannot both be right and neither is listed in `CLAUDE.md`'s authoritative set. `metrics_formal.md` was modified today (2026-08-11) and is the newer, more formal treatment. **Merge into one, keep the survivor, then delete the other — never delete before merging.** |
| `docs/pns_metric_design_results.md` (35 lines) | `[UNTRACKED — IRREVERSIBLE]` | Its PN/PS numbers now also live in `05_evaluation_plan.md` §5 and `general_plan.md` §7, and they are n = 3 numbers that P1-1 will supersede. Fold the surviving content into `05_evaluation_plan.md`, then delete. |
| `docs/04_literature_gaps.md` + `general_plan.md` §3 + `docs/BenchmarkingTSCFEs.md` | mixed | Three related-work statements. `general_plan.md` §3 is authoritative by rule. Reduce to two (authoritative claim + verified bibliography); delete the third per §12.1. |
| `docs/reviews/m2_milestone_review.md` | `[UNTRACKED — IRREVERSIBLE]` | Substantially superseded by `docs/archive/m2_multiseed_and_pearl_carla.md` and by `DECISIONS.md`. Low cost to keep; if space matters, fold into `DECISIONS.md` first. |

### 12.3 KEEP — provenance, do not delete

| Path | Why it must survive |
|---|---|
| `docs/archive/` (all four files + `plans/`) | `CLAUDE.md` and `docs/archive/README.md` both state these are retained for provenance and must not be cited. **All `[UNTRACKED — IRREVERSIBLE]`.** They are the only record of what was believed, when, and on what evidence — including the `d3c377d` CF-faith retraction. |
| `docs/archive/m2_multiseed_and_pearl_carla.md` | ⚠️ **IRREPLACEABLE.** The only surviving record of the 2026-07-30 decision that lowering `lam_prox` cannot rescue PearlCARLA because the intervention decays below float32 precision. That decision is **load-bearing for Finding 4's diagnosis in this memo** — without it, someone will re-propose the tuning fix and waste a sprint. Do not delete under any circumstances. |
| `docs/archive/m4_ablation_presets_smoke.md` | ⚠️ **IRREPLACEABLE.** `archive/README.md` explicitly records that its *design* content is still cited from live source — `benchmarks/generator.py`, `benchmarks/mechanisms.py`, `config.py`, `tests/test_config.py`. It is also now the design record for a preset (`SMOKE_NONMONOTONIC`) this memo shows to be vacuous; deleting it destroys the record of what was intended, which is what makes the failure diagnosable. |
| `docs/spec_code_reconciliation.md` | ⚠️ **IRREPLACEABLE.** `CLAUDE.md` states the operator dictionary `Φ` "survives as specification in `docs/spec_code_reconciliation.md`; revive it from there, not from git history" — after the `scm/` cluster deletion of 2026-08-03. `[UNTRACKED]`, so there *is* no git history to revive from. Its mis-titled "(R10)" heading is a cosmetic wart, not a reason to delete. |
| `DECISIONS.md` (2 156 lines) | Chronological decision log, `[UNTRACKED — IRREVERSIBLE]`. Length is not staleness; a decision log is *supposed* to grow. Do not prune. |
| `ROADMAP.md`, `PROGRESS_DASHBOARD.md`, `PROJECT_GOVERNANCE.md`, `AGENT_CHARTERS.md`, `CLAUDE.md` | Governance set, all `[UNTRACKED — IRREVERSIBLE]`, all named as authoritative by `CLAUDE.md`. Keep. |
| `docs/risk_register.md`, `docs/axis_metrics_report.md`, `docs/method_provenance.md`, `docs/cf_faith_methodology.md`, `docs/figure_style.md`, `docs/05_evaluation_plan.md`, `docs/06_publication_targets.md`, `docs/general_plan.md` | `CLAUDE.md`'s named authoritative set. All `[UNTRACKED]`. Keep. |
| `docs/references_verified.md`, `docs/pns_metric_design.md` | `[TRACKED]`, authoritative. Keep. |

### 12.4 Code / config, pending a recorded PI decision

Per `CLAUDE.md`, removing a preset is a scope change requiring a decision first.
Flagged, not recommended for immediate action:

- `SMOKE_NONMONOTONIC` (`config.py:316`) — the ablation it exists for is vacuous.
  Either it becomes meaningful after P0-2 (likely — a genuinely curved mechanism
  makes `sin` vs `tanh` a real perturbation) or it should be retired. **Decide
  after P0-2, not before.**
- `FULL_INTERIOR_LABEL` (`config.py:257`) — its own docstring says it is
  superseded by `FULL_INTERIOR_LABEL_LATE` and that its classifier trains to
  0.501 (chance). Retained deliberately so the negative result stays
  reproducible. **Keep**, and leave the docstring as the record.

---

## 13. Bottom line

The instrument is sound and the negative finding about explainer behaviour
(`D`, `frac_vacuous`) is real, novel, and worth publishing. The evidence base
underneath it is not currently in a state that would survive review: one of two
mechanism families is not what it claims to be, one of four discovery columns is
a library convention printed as a measurement, one method row is unscorable, one
metric column is sign-inverted against its own specification, one method is
counted twice, and no headline meets the project's own pre-registered seed count.

None of that is fatal. All of it is fixable, and the fix list is short and
concrete (§11). **A killed tier in week 2 is a gift; a killed paper in review is
a wasted year.** The correct move is to withdraw the nonlinear-generalisation
claim now, narrow §2 to what the linear tier supports (§10), fix the mechanism,
and re-run at the seed count the pre-registration already committed to.

What must not happen is the tempting alternative: keeping `full_nl` in the tables
because the numbers "look fine". They look fine because they are the linear
numbers computed twice.
