# Spec ↔ Code Reconciliation (R10)

**Date:** 2026-07-14 · **Trigger:** PI review action (R10) — three definitions in
`docs/updated_general_plan.md` contradicted the implemented code. Per the PI
directive, each divergence is resolved to **one source of truth, chosen
deliberately**, and `updated_general_plan.md` is edited to describe the
implemented reality so the eventual methods section is reproducible.

**Governing principle.** For a *benchmark* paper the source of truth must be the
code that actually produces the reported numbers — i.e. the **wired, tested**
pipeline (`config.py` presets → `data_io` → `experiments/01–08` → `results/`).
Where a spec-matching implementation exists but is **dormant** (not wired into
any config/experiment, no tests), it cannot be canonical without wiring +
validation work that is out of scope here; it is documented as a dormant
alternative, not silently presented as the method.

---

## 1. Mechanism family (Eq. 1)

| | |
|---|---|
| **Spec (drafted)** | `X_t^{(j)} = Σ w_ij · φ_ij(X_{t−τ}^{(i)}) + U`, with `φ_ij` sampled from the invertible operator dictionary `Φ = {identity, sin, cos, tanh, |·|, (·)², exp(−|·|)}`. |
| **Code — LIVE** | Two wired families: **LinearSCM-T** (`LinearSCMT`, VAR, i.e. `φ = identity` — a *special case* of Eq. 1, fully consistent); **NlinearSCM-T** (`NlinearSCMT` / `MLPMechanism`): additive-noise per-node MLP `X_t^i = decay_i·X_{t−1}^i + gain·tanh(MLP_i(masked lagged parents)) + U_t^i`, spectral-norm-capped. Used by every config, experiment, result, and test. |
| **Code — DORMANT** | `causaltemp_xai/scm/operators.py` implements the exact operator dictionary `Φ` (+ `sample_mechanism`, per-edge `Mechanism`). **Unwired** (no config/data_io/experiment uses it) and **untested** (no `tests/test_scm*`). |

**Decision: canonical = the LIVE additive-noise ANM (linear + MLP instantiations).**
Rationale: Eq. 1 is really an umbrella additive-noise structural equation
`X_t^j = f_j(pa(X_t^j)) + U`; the linear case is `φ=identity`, and NlinearSCM-T's
per-node MLP over masked parents *subsumes* a sum of per-edge operators while
preserving **additive noise** (which is what makes Pearl abduction the exact
subtraction the whole CF-faith/oracle machinery relies on). The per-edge
operator-dictionary form is a legitimate alternative parameterisation but is
dormant.

**Spec edit:** rewrite Eq. 1 as the additive-noise ANM with its two implemented
instantiations; move the operator dictionary `Φ` to a footnote flagged as the
non-monotonic ablation's operator source (the step function `𝟙[·>0]` is drawn
from this pool) and as a dormant alternative parameterisation, not the primary
mechanism.

---

## 2. ICC (Axis A)

| | |
|---|---|
| **Spec (drafted)** | Latent-traversal flip-rate: `ICC_i = (1/N) Σ_n 𝟙[ f(D_ψ(z^n + δ_i e_i)) ≠ f(x^n) ]` — perturb one latent axis, decode, count label flips. Requires an encoder/decoder `D_ψ`. |
| **Code — LIVE** | `axis_a.icc(attribution, int_channel)` — a **decoder-free** proxy: fraction of a saliency map's attribution mass on the causally-relevant channel. Wired via `compute_axis_a` (used in `experiments/_common.py`). |
| **Code — matches spec but DORMANT** | `axis_a.icc_latent(encoder, decoder, ...)` implements the spec formula *exactly* (perturb latent, decode, flip-rate). Exported, but **not referenced by any experiment or test**. |

**Decision: keep BOTH as two operationalisations of one concept, with the
latent-traversal `icc_latent` as the *definitional* form.** Rationale: the spec's
`D_ψ`-based ICC only applies to methods that *have* an encoder/decoder (concept /
representation learners — iVAE, CITRIS, CBM-T); attribution methods (TimeSHAP,
Dynamask, IG) have no decoder, so a decoder-free proxy (`icc`) is needed to score
them at all. These are complementary, not contradictory.

**Spec edit:** keep the latent-traversal formula as the ICC definition (it is
implemented as `icc_latent`), and add a sentence that attribution methods —
which lack `D_ψ` — are scored with a decoder-free attribution-mass proxy
(`icc`). These are complementary operationalisations, not to be merged.

**Correction + wiring (2026-07-15, after PI metric review).** `icc_latent` was
reviewed and **corrected** before wiring — the drafted formula's baseline was
methodologically wrong:

- **Baseline deviation from the drafted spec [definitional].** The spec wrote
  `f(x^n)` (prediction on the *original input*) as the baseline. That conflates
  the decoder's reconstruction error with the traversal effect: a lossy `D_ψ`
  flips labels even at `δ=0`, inflating *every* `ICC_i`. The corrected metric
  uses the **matched reconstruction baseline**
  `ICC_i = mean( f(D_ψ(z+δe_i)) ≠ f(D_ψ(z)) )`, isolating the traversal. This is
  a deliberate, documented deviation from the drafted `f(x^n)`.
- Also corrected: per-dimension **std-scaled** step `δ_i = c·std(z_i)` (a raw
  `δ` confounds causal relevance with each latent's scale), **±δ averaging**,
  and a deterministic decode. Unit-tested in `tests/test_icc_latent.py`
  (high ICC for the causal dim, ~0 for others, baseline absorbs recon noise).
- **Wired**: `experiments/07_auxiliary_methods.py` runs it on the decoder-based iVAE,
  with the PI-mandated guards — a **reconstruction-fidelity gate**
  (`recon_label_agreement ≥ 0.8`), **latent→factor Hungarian alignment** (MCC)
  before the parent-vs-non-parent contrast, a pre-registered magnitude ladder
  `c∈{1,2,3}`, and a permuted-assignment null. **Honest finding at smoke scale:**
  the iVAE fails the reconstruction gate (`recon_label_agreement ≈ 0.5`,
  posterior-collapse-like / bottleneck failure) even with KL-warmup and low
  `β`, so ICC is *not interpretable via iVAE on this benchmark* — reported as-is,
  not forced. This corroborates the open identity-mixing / representation-method
  concern (Axis A has little signal here). A stronger temporal decoder (or the
  nonlinear-mixing benchmark variant) is the path to an interpretable ICC.

---

## 3. CF-faith (Axis C)

| | |
|---|---|
| **Spec (drafted)** | Normalised DTW to the analytical oracle CF: `CF-faith = 1 − DTW(X'_exp, X'_CF) / DTW(X, X'_CF)`; hard = `𝟙[DTW(X'_exp, X'_CF) < ε]`. |
| **Code — LIVE** | `metrics/cf_faith.py::CFfaith` — **residual-based SCM-consistency**: `soft = exp(−residual/scale)`, `hard = 𝟙[no retroactive pre-intervention change AND forward mechanism-residual < tol]`, under **two semantics** (`noiseless_rollout`, `pearl_delta`). No DTW. This is the tested, load-bearing metric the entire pipeline (oracle CFs, Phase 04/05/08, the rollout-vs-pearl contrast) is built on. |

**Decision: canonical = the residual SCM-consistency CF-faith (two semantics).**
Rationale: it is strictly more general than DTW-to-a-single-oracle — it tests
whether a CF is consistent with the *mechanism* via abduction–rollout, which (i)
does not require committing to one oracle trajectory, and (ii) yields the
`noiseless_rollout` vs `pearl_delta` distinction that is central to the
benchmark's headline finding. Replacing it with DTW would discard that and break
every downstream phase.

**Spec edit:** replace the DTW definition with the residual/two-semantics
definition as implemented.

**Update (2026-07-15) — DTW discarded outright.** The earlier resolution kept
DTW-to-oracle alive as an optional *complementary* CF-faith number. That option
is now **withdrawn** and every DTW formulation is removed from the repo (spec
and code). Rationale: offering a second, weaker CF-faith invites exactly the
ambiguity the two-semantics split exists to eliminate — DTW measures distance
to *one privileged oracle trajectory*, which (i) re-introduces the commitment
the residual formulation was designed to avoid, and (ii) cannot express the
rollout-vs-Pearl distinction at all. Removed: the DTW definition here and in
`updated_general_plan.md` §Axis C, and `axis_c.proximity_dtw` (a normalised DTW
proximity helper that was exported but **unwired** — no experiment called it —
and which pulled in an optional `tslearn` dependency). No results change: no
reported number ever came from it.

---

## 4. IVR retired; TRSI reframed (2026-07-15)

Two Axis-C entries were re-examined after the metric review and resolved by
**removal** and **reframing** respectively. Both were reported alongside
CF-faith in a way that overstated what they measure.

### 4.1 Irreversibility-Violation-Rate — **removed**

| | |
|---|---|
| **Was** | `axis_c.ivr(X_cf, X, T_int)` — fraction of CFs editing any `t < T_int`; wired per-instance in `experiments/_common.py`, aggregated, and reported in `eval_*.json` / `per_instance.csv` / seed tables. |
| **Now** | Deleted from code, spec, tests, and the seed-aggregate schema. |

**Decision: retire it.** Two independent reasons, either sufficient:

1. **Non-discriminative by construction in the live wiring.** The pipeline
   derives `T_int` via `derive_intervention_t` using `INTERVENTION_TOL` — the
   *same* per-element predicate IVR then tests against. So IVR is **0 by
   construction** for every method the benchmark actually scores. It was a
   pipeline-consistency canary reported in a column that reads as a
   discriminative score. (This was already conceded in its own docstring.)
2. **Redundant where it *would* discriminate.** For a method declaring its own
   `T_int`, the property IVR checks — no retroactive pre-intervention edit — is
   exactly clause (i) of CF-faith's retro gate, which zeroes *both* CF-faith
   scores on violation. CF-faith subsumes it.

**Regression coverage is preserved.** The CELS false-flag tests
(`tests/test_metric_adversarial.py::TestCelsFalseFlagRegression`) asserted the
M1 tolerance fix through both IVR *and* CF-faith; the CF-faith assertions —
including `test_genuine_retroactive_edit_still_flagged` — retain full coverage
of the retro gate and the shared `INTERVENTION_TOL` contract. Only the
duplicate IVR asserts were dropped.

**Consequence for P3.** The normative-principles table now maps P3 to
"OOD plausibility; CF-faith's retroactive gate" — the "Time Traveler Dilemma"
is still foreclosed, by the gate rather than by a standalone metric.

### 4.2 TRSI — **kept, reframed**

TRSI stays (it is the only Axis-C column sensitive to *temporal* edit
incoherence, and unlike CF-faith it needs no mechanism, so it transfers to real
data). What changed is the claim attached to it. It is now documented — in
`axis_c.trsi`, `updated_general_plan.md`, and the test module — as:

* **adopted, not novel** (ported from `causal_tscf_bench`);
* a **mechanism-free heuristic proxy**, never evidence of causal faithfulness —
  it does not consult the SCM, and while a mechanism-consistent CF is generally
  TRSI-smooth, **the converse does not hold**;
* a **descriptor with no ground-truth optimum** — a genuinely abrupt
  intervention *should* score high, so it is read against the other Axis-C
  columns rather than minimised;
* subordinate to CF-faith, which answers the propagation question (P2) directly.

Its adversarial test contract was narrowed to match: TRSI is asked only to
*order* edits by temporal coherence, not to flag a causal violator.

---

## 5. Shift-VR denominator bug + structured sparsity (2026-07-15)

### 5.1 Shift-VR's `validity_base` measured the wrong counterfactuals

**Found while investigating why Phase 03 was so slow.** `shift_vr` regenerated
the base CFs internally (`_generate_batch(method, X_base_test, ...)`), even
though Phase 03 had *just* generated exactly those arrays for exactly those
inputs (`X_base_test` **is** `X_sel`) and persisted them to
`cf/X_cf_<Method>.npy`. Two consequences:

1. **Cost.** The phase paid for every method's recourse twice — 3N
   method-generations where 2N suffice (N saved + N redundant base + N shift).
   CF generation dominates the phase, so this was the bulk of the "forever".
2. **Correctness (the serious one).** `_generate_batch` is only deterministic
   for deterministic methods. **`CftsCounts` is not**: regenerating twice on
   identical inputs gives validity **0.40** and **0.30**, while the persisted
   array scores **0.60**. So Shift-VR's denominator described a CF set that
   *nothing else in the pipeline ever saw*, and that **contradicted the
   validity Phase 04 reports for the same method+config**.

**Fix:** `shift_vr` takes an optional `cf_base={name: array}`; Phase 03 passes
the arrays it just generated and persisted. The denominator is now *the same
object* Phase 04 scores. The numerator (`validity_shift`) is still a fresh
generation — that is the metric's actual signal — so Shift-VR stays noisy for
stochastic methods; the `MIN_VALIDITY_BASE_FOR_RATIO` guard and multi-seed CIs
are what control that.

**Measured:** 2.00× on `shift_vr` (generation calls 2→1 per method), verified
bit-identical for deterministic methods. Phase 03 `smoke --n-cf 10`: **6m26s**.

**Impact on existing numbers.** Every previously reported `shift_vr` /
`validity_base` for a **stochastic** method is suspect and must be re-run;
deterministic methods are unaffected (6 of 7 smoke methods were unchanged;
only `CftsCounts` moved: `validity_base` 0.5→0.6, matching its persisted
array). **Any H4/robustness claim resting on the old Shift-VR table is
provisional until re-run.** Regression-tested in
`tests/test_metric_adversarial.py::TestShiftVRBaseReuse`.

**Adjacent latent bug fixed.** A method that raised during Phase 03's
generation loop was caught and logged, then passed to `shift_vr` anyway, where
it raised **uncaught** — killing the phase *after* all the expensive work. Such
methods are now excluded from the Shift-VR set (they cannot yield a base
validity regardless) and reported as skipped.

### 5.2 Structured sparsity wired

`axis_c.sparsity(..., return_detailed=True)` (channel/timepoint sparsity)
existed in `HEAD` but was **dormant** — every caller used the flat scalar
branch, so no reported number ever came from it. Now wired through
`eval.evaluate_method`, `experiments/_common.py`
(`per_instance_records` / `aggregate_method_row` / `SEED_AGGREGATE_METRICS`),
and the Phase 04 summary table as `sparsity_channels` / `sparsity_timepoints`.

**Why it earns its place:** flat sparsity is blind to the *shape* of the edit.
Verified on constructed edits with **identical flat sparsity (0.80)** and
exactly inverted structure — "one channel, all timesteps" gives
`(ch=0.80, tp=0.00)`, "few timesteps, all channels" gives `(ch=0.00, tp=0.80)`.
High `sparsity_channels` = "few variables"; high `sparsity_timepoints` =
"few moments". This may support a sharper hypothesis about which *shape* of
edit each method paradigm produces.

---

## Summary of resolutions

| Divergence | Source of truth | Action on code | Action on spec |
|---|---|---|---|
| Eq. 1 mechanism | LIVE additive-noise ANM (linear + MLP) | none (dormant `scm/operators.py` kept, flagged) | rewrite Eq. 1 to the ANM + instantiations; `Φ` → footnote |
| ICC | latent-traversal (definitional) + attribution proxy | corrected (matched baseline, ±δ, std-scaling) + wired in Phase 07 | keep formula; document the baseline deviation + attribution proxy |
| CF-faith | residual SCM-consistency, two semantics | none | replace DTW definition with residual definition |
| DTW (§3 update) | — (discarded) | remove `axis_c.proximity_dtw` (unwired) | remove the DTW CF-faith definition and the "complementary metric" offer |
| IVR (§4.1) | — (retired) | remove `axis_c.ivr` + all wiring/tests/schema | remove the metric; P3 → CF-faith retro gate |
| TRSI (§4.2) | LIVE `axis_c.trsi` | none (docstring reframed) | reframe as adopted, mechanism-free proxy; drop the novelty claim |

The first three required no code change — the live code stood and the spec was
corrected to match. The last three are the 2026-07-15 metric-review outcomes:
DTW and IVR are **deleted** (neither ever produced a reported number that
survives, and IVR was 0-by-construction in the live wiring), and TRSI's *claim*
is corrected without touching its computation. The earlier "optionally add
DTW-to-oracle" P1 is **withdrawn**, not deferred; the "wire `icc_latent`" P1 is
**done** (Phase 07). Spec and code now tell one consistent story, clearing the
methods-section gate.
