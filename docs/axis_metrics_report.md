# Axis Metrics Report — Implementation vs. Obtained Results

**Date**: 2026-07-02 · **M1 resolution pass**: 2026-07-07 (all "fix recommended" rows closed — see §9)
**Scope**: `causaltemp_xai/metrics/{axis_a,axis_b,axis_c,axis_d,cf_faith}.py` as exercised by the
phased experiment pipeline (`experiments/01`–`06`).
**Purpose**: document precisely how each axis is computed, relate that to the numbers the
pipeline actually produced, and flag anything that looks like an implementation issue rather
than a real signal — so it's clear what to trust and what to fix.

> **M1 note (2026-07-07)**: the numbers quoted in §1–§7 below are the *audited* (pre-fix,
> `n_cf=10`) values that motivated the fixes and are kept for the audit trail. The regenerated
> post-fix tables (`n_cf=20`, fixed metrics) live under `results/`; the resolution log in §9
> records what changed and why.

**Data used in this report** (all runs are reproducible via the commands in `results/README.md`):

| Config | Mechanism | Classifier | Test acc | n_cf | Methods |
|---|---|---|---|---|---|
| `smoke` | linear VAR(1), k=5, T=30 | LSTM (fully trained) | 92.0% | 10 | CARLA, CftsWachter, CftsCOMTE, CftsConfeti, CftsCounts, CftsCels |
| `smoke_nl` | additive-noise MLP, k=5, T=30 | LSTM (fully trained) | 95.0% | 10 | same 6, plus classifier-free OracleCF-Pearl/Rollout |

Raw artifacts: `results/smoke/lstm/summary.json`, `results/smoke_nl/lstm/summary.json`,
`results/smoke_nl/oracle/summary.json`, `results/tables/table_axis_c_cf_faith.csv`,
`results/{smoke,smoke_nl}/lstm/per_instance.csv`.

---

## 1. CF-faith (`metrics/cf_faith.py`) — the benchmark's hinge metric

### How it works

`CFfaith.score(x_original, x_cf, intervention_t, graph, mechanism)` does two things:

1. **Retroactive check**: sums `|x_cf[t] - x_original[t]|` over **all** `(t, channel)` pairs for
   `t < intervention_t`. If that sum exceeds `tol` (default `1e-4`), the CF is immediately
   `hard=0, soft=0` — no forward simulation is even attempted. This encodes "a counterfactual may
   not retroactively rewrite the past."
2. **Forward-consistency check** (only if step 1 passes), in one of two semantics:
   - `noiseless_rollout`: roll `x_cf` itself forward through the mechanism with **no noise**
     from `intervention_t+1` onward, and compare to the CF's own claimed values. Faithful iff the
     CF *is* a pure deterministic SCM continuation.
   - `pearl_delta`: abduct the exogenous noise from the **factual** trajectory
     (`eps[t] = x_orig[t] - mechanism.forward_numpy(window)`, exact because noise is additive),
     then roll forward reusing that noise. Faithful iff the CF matches Pearl's
     abduction-action-prediction counterfactual.
   Both produce an L1 residual `r`; `hard = 1[r < tol]`, `soft = exp(-r / scale)`.

Both semantics are always computed and reported side by side (`cf_faith_rollout_*` /
`cf_faith_pearl_*`) — the plan's explicit "keep both metrics" decision, because a method built to
satisfy one will generally score near-zero on the other, and that gap *is* the benchmark result.

### What we obtained

| Method | rollout_hard | pearl_hard | Interpretation |
|---|---:|---:|---|
| **CARLA** (smoke) | **1.00** | 0.00 | Built to satisfy noiseless rollout — does, by construction |
| **CARLA** (smoke_nl) | **1.00** | 0.00 | Same result holds on the nonlinear mechanism |
| CftsWachter | 0.00 | 0.00 | Gradient CF, ignores the SCM entirely — as expected |
| CftsCOMTE | 0.00 | 0.00 | Shapelet-replacement, non-causal — as expected |
| CftsCounts | 0.00 | 0.00 | VAE-based, non-causal — as expected |
| CftsCels | 0.00 | 0.00 | See §5 — but for a different reason than "not causal" |
| CftsConfeti | 0.70 (smoke) / 0.40 (smoke_nl) | same values | Partial faithfulness — see below |
| OracleCF-Pearl (smoke_nl) | 0.00 | **1.00** | Faithful under the semantics it's built for |
| OracleCF-Rollout (smoke_nl) | **1.00** | 0.00 | Faithful under the semantics it's built for |

**This is exactly the benchmark's core finding, reproduced cleanly on both mechanisms**: only the
method whose objective is literally "roll the mechanism forward" (CARLA, and by construction the
oracle) is causally faithful; every off-the-shelf CF explainer (COMTE, Wachter, Counts) achieves
`validity=1.0` while being causally arbitrary (`rollout_hard=0.0`). High classifier-flip validity
does not imply causal faithfulness — the whole point of the benchmark.

**CftsConfeti is the interesting middle case**: `validity=0.30–0.60` but `rollout_hard=0.40–0.70`.
It makes very small, sparse edits (`sparsity=0.69–0.85`, i.e. 69–85% of the 150 `(T,k)` entries
are untouched) — small enough that they frequently land *inside* the mechanism's own tolerance for
"a plausible continuation," without the method ever consulting the mechanism. This is a real,
interesting signal (genetic search sometimes finds small, accidentally-faithful edits), not a bug.

### Implication for debugging

If a future method claims high faithfulness, check `sparsity`/`proximity_l1` alongside it —
CftsConfeti shows that near-zero edits can accidentally pass the faithfulness check without any
causal reasoning. A method should only be credited as "faithful" if it also clears a meaningful
`validity`/`proximity` bar; faithfulness alone is gameable by making (almost) no change.

> **RESOLVED (M1, 2026-07-07) — joint faithfulness–validity criterion implemented.** The metric
> suite now reports `cf_faith_rollout_hard_valid` and `cf_faith_pearl_hard_valid`: the fraction of
> instances that are **both** hard-faithful **and** classified as the target class, computed
> per-instance as `hard_i × 1[f(x_cf_i) = target]` in `eval.evaluate_method`,
> `experiments/_common.per_instance_records`, `aggregate_method_row`, and the accumulated
> `results/tables/table_axis_c_cf_faith.csv`. A tiny-edit CF can still pass the faithfulness check
> in isolation (both marginals are still reported), but it earns **zero joint credit** unless it
> also flips the classifier — closing the loophole where "faithful" could be claimed while flipping
> nothing. Design note: no *proximity* floor is included, deliberately — proximity should be small,
> and a CF that is tiny AND valid AND faithful is a genuinely excellent CF, not a gamed one; the
> loophole exists only when validity is not required. Tests:
> `tests/test_metric_adversarial.py::TestJointFaithValidCriterion`.

---

## 2. Axis C (`metrics/axis_c.py`) — counterfactual quality

### How each metric works

| Metric | Formula | Range | Notes |
|---|---|---|---|
| `validity` | fraction of CFs the classifier assigns to `target_class` | [0,1] | pure flip-rate |
| `proximity_l1/l2` | `‖x - x_cf‖_1` / `‖x - x_cf‖_2` over the flattened `(T,k)` | [0,∞) | lower = closer |
| `sparsity` | fraction of `(T,k)` entries with `|Δ| ≤ 1e-6` | [0,1] | **1 = unchanged**, so higher sparsity + high validity is the ideal combo |
| `OOD` | `IsolationForest.decision_function` fit on `X_train`, scored on the CF | (-∞,∞) | positive = in-distribution, negative = anomalous |
| `TRSI` | mean L2 norm of the **second difference** of `Δ_t = x_cf_t - x_t` over time | [0,∞) | measures whether edits are temporally smooth vs. jumpy |
| `IVR` | fraction of instances with `max|Δ|` for `t < intervention_t` exceeding `1e-4` | [0,1] | 0 = no retroactive edits |

### What we obtained

| Method (smoke) | validity | prox_l1 | sparsity | OOD | TRSI | IVR |
|---|---:|---:|---:|---:|---:|---:|
| CARLA | 1.00 | 34.3 | 0.41 | 0.084 | 0.60 | 0.00 |
| CftsWachter | 1.00 | 64.5 | 0.00 | 0.056 | 1.38 | 0.00 |
| CftsCOMTE | 1.00 | 51.5 | 0.00 | −0.047 | 0.38 | 0.00 |
| CftsConfeti | 0.30 | 8.1 | 0.85 | 0.058 | 0.28 | 0.00 |
| CftsCounts | 0.60 | 40.2 | 0.00 | 0.108 | 1.00 | 0.00 |
| CftsCels | 1.00 | 2.0 | 0.05 | 0.063 | 0.055 | **1.00** |

### Implications, sanity checks, and one real finding

- **CftsWachter has both the worst proximity (64.5) and the worst TRSI (1.38)** of the non-causal
  methods — makes sense, it's an unconstrained gradient method with no smoothness penalty in this
  configuration; it changes every one of the 150 entries (`sparsity=0.0`).
- **CftsCOMTE's OOD is negative** (−0.047 smoke, −0.048 smoke_nl) — the only method the
  IsolationForest flags as *out*-of-distribution on average. Its shapelet-replacement swaps in
  segments from other training instances wholesale, which is plausible to produce distributional
  mismatch; this tracks with intuition and isn't a metric bug.
- **CftsCels has by far the smallest proximity (2.0 / 0.57) yet `IVR = 1.00` — every single
  instance is flagged as retroactively edited.** This looked suspicious enough to investigate
  directly (see §5): it is a genuine implementation-interaction issue, not a real causality
  violation, and it also explains why CftsCels's CF-faith is exactly `0.0` (not just `hard=0`, but
  `soft=0.0` too) in §1's table.
- **`sparsity` and `frac_altered` are complements** (`frac_altered = 1 - sparsity`) and both are
  reported in the JSON — redundant but harmless; kept for direct comparability with
  `causal_tscf_bench`'s schema.

---

## 3. Axis D (`metrics/axis_d.py`) — robustness

Two sub-metrics are wired into the pipeline, applied to what they're actually suited for (not to
every method — see the routing rationale in `experiments/_common.py`):

### Shift-VR (Phase 03, all CF methods)

`shift_vr(model, methods, X_base_test, X_shift_test, ...)`: keep the **frozen classifier**, run
each CF method fresh on both a base test set and one drawn from a noise-distribution-shifted
version of the same SCM (`shifted_config`, Laplace→Uniform noise, same graph/mechanism by
construction), and report `validity(shift) / validity(base)`.

| Method | smoke shift_vr | smoke_nl shift_vr |
|---|---:|---:|
| CARLA | 1.00 | 1.00 |
| CftsWachter | 1.00 | 1.00 |
| CftsCOMTE | 1.00 | 1.00 |
| CftsCels | 1.00 | 1.00 |
| CftsCounts | 1.00 | 0.90 |
| CftsConfeti | **2.67** | 1.17 |

Most methods are stable (≈1.0). **CftsConfeti's `shift_vr = 2.67`** on `smoke` comes from
`validity_base = 0.30` (3/10) → `validity_shift = 0.80` (8/10) — a 3.7x swing on a denominator of
only 3 successes. This is a **small-sample artifact of `n_cf=10`**, not a real robustness
improvement under shift; the ratio metric is numerically unstable whenever `validity_base` is
small. **Recommendation**: don't trust `shift_vr` ratios computed from `validity_base < ~0.3` at
this sample size; re-run with `n_cf≥50` before drawing conclusions about CftsConfeti specifically.

> **RESOLVED (M1, 2026-07-07).** `eval.shift_vr` now enforces exactly this recommendation in code:
> the ratio is reported **only when** `validity_base ≥ MIN_VALIDITY_BASE_FOR_RATIO = 0.3`
> (otherwise `None`/JSON `null` — which also removes the invalid bare-`NaN` JSON sentinel), and
> the `(validity_base, validity_shift)` pair plus `n_base`/`n_shift` are **always** reported so
> no information is lost. Both options offered by the M1 task ("floor" vs "pair") are effectively
> implemented — the floor gates the *ratio*, the pair remains the primary record. Tests:
> `tests/test_metric_adversarial.py::TestShiftVRGuard` (suppression below floor, inclusive
> boundary at 0.3, `None` — not `NaN` — at zero base, pair always present).

### Input-sensitivity (Phase 03, IG attribution only)

`input_sensitivity(X, attribution_fn, eps=0.01)`: perturb the input with small Gaussian noise,
recompute the IG saliency map, report mean relative L2 change.

| Config | InputSens |
|---|---:|
| smoke (linear) | 0.019 |
| smoke_nl (nonlinear) | 0.133 |

**IG's saliency map is ~7x more sensitive to small input perturbations on the nonlinear
mechanism than the linear one.** This is a plausible, real signal: IG's path-integral gradient
follows a straight line in input space between baseline and input, and a classifier trained on
data generated through a nonlinear (tanh-bounded, MLP) SCM likely has a less smooth decision
boundary than one trained on linear-VAR data, so small input jitter moves the local gradient more.
Worth double-checking if it recurs at `full_nl` scale — if it grows further, it may indicate the
nonlinear classifier itself is less well-regularized rather than a metric issue.

### Not wired in this pipeline

`concept_stability` (Axis D's third sub-metric) needs a `concept_fn` — no concept-extraction
method exists in this codebase, so it's correctly left `None`/omitted rather than faked.

---

## 4. Axis A (`metrics/axis_a.py`) — attribution/concept quality

Only the "bench-ported" functions (`icc`, `mcc_concept`, `compute_axis_a`) are exercised here —
the identifiability-theory functions (`icc_latent`, `mig`, `dci`, `mcc`) need an encoder/decoder
or latent factors that don't exist in this pipeline (no concept-based XAI method is implemented),
so `LD`/`MCC_disent` are correctly reported as `NaN` rather than fabricated.

### How it works

For each of the `n_cf` selected instances, `experiments/_common.build_oracle_interventions`
builds a **ground-truth** `do(x[t0, node]=value)` via `structural_counterfactual` (cycling
`node = i % k` across instances) — this is real ground truth, not a proxy, and it works for
both `LinearMechanism` and `MLPMechanism` since `structural_counterfactual` is mechanism-generic.
Then, on the IG saliency map for the **factual** instance:

- `icc(attribution, int_channel)` = fraction of `Σ|attribution|` that lands on the true
  intervened channel's column.
- `mcc_concept(attribution, causal_parents)` = fraction of `causal_parents(node)` whose total
  attribution exceeds a fixed threshold `1e-3`. Instances with **zero** causal parents (a root
  node) are excluded from the mean (`NaN`, filtered before averaging) — correct, since
  "coverage of zero parents" is undefined, not zero.

### What we obtained

| Config | ICC | MCC_coverage | # instances (of 10) with ≥1 parent |
|---|---:|---:|---:|
| smoke | 0.216 | 1.00 | 8 (2 instances hit the k=5 graph's parentless node) |
| smoke_nl | 0.198 | 1.00 | 8 |

### Implications — two things worth flagging

1. **ICC ≈ 0.20–0.22 is barely above chance.** With `k=5` channels, a saliency map with attribution
   *uniformly* spread across channels would score `ICC ≈ 1/5 = 0.20` by definition. IG is not
   meaningfully concentrating attribution mass on the channel that was (in the oracle's synthetic
   sense) causally intervened on. This is **not necessarily a metric bug** — IG explains the
   *classifier's* decision boundary on the *factual* instance, which has no relationship to the
   oracle intervention we retroactively paired it with for evaluation purposes. The metric is
   correctly implemented; the *interpretation* needs care: this measures "does IG happen to point
   at the same channel the benchmark's oracle would intervene on," not "is IG faithful to the
   classifier." Near-chance is a plausible, even expected, null result here.
2. **MCC_coverage = 1.00 exactly, on both configs, is a ceiling effect from a weak threshold.**
   `mcc_concept`'s `threshold=1e-3` (default, never overridden by the pipeline) is very low relative
   to typical IG attribution magnitudes — dense/noisy saliency maps clear it almost everywhere by
   default, so "coverage" is trivially satisfied. **This metric currently has near-zero
   discriminative power** at the default threshold; it cannot currently distinguish a genuinely
   causally-aligned attribution method from a random one. **Recommended fix**: either raise the
   threshold to something relative to the attribution map's own scale (e.g. `threshold =
   0.05 * max(|attribution|)` per instance) or report the raw per-channel attribution mass instead
   of a thresholded binary coverage flag.

   > **RESOLVED (M1, 2026-07-07) — continuous-mass formulation adopted.** `mcc_concept` is now the
   > **chance-normalized causal mass**: (share of total `|attribution|` mass on parent channels) ÷
   > (`|Pa|/k`, the share a uniform map would place there). `1.0` = chance, `>1` = concentrates on
   > parents, `<1` = avoids them; max `k/|Pa|`; `NaN` for zero parents or a zero-mass map. Why not
   > the scale-relative *threshold* option: it fixes scale-dependence but **not** the ceiling —
   > dense IG-like maps still put non-trivial mass on every channel and clear any sane per-channel
   > threshold, so thresholded coverage stays pinned at 1.0. The continuous form is threshold-free,
   > scale-invariant by construction, and mirrors the `icc` attribution-mass formulation already
   > used on this axis. Tests: `tests/test_metric_adversarial.py::TestMCCCoverageFix` (parent-
   > avoiding map → 0, concentrated → k/|Pa|, uniform → exactly 1.0, scale invariance, no ceiling
   > for dense maps, NaN edge cases). Semantics change note: the reported quantity is no longer a
   > "fraction of parents covered" — results tables keep the `MCC_coverage` key, and the paper must
   > describe the mass definition.

---

## 5. Axis B (`metrics/axis_b.py`) — graph diagnostic (dataset-level, not per-method)

### How it works and why it's dataset-level here

No graph-*discovery* method is wired into this pipeline (nothing infers a causal graph from data),
so `axis_b_benchmark_diagnostic` (in `experiments/_common.py`) calls `compute_axis_b(adj, adj, X=X_test)`
with the **ground-truth graph as both arguments** — `SHD=0` and `LagAcc=1.0` are therefore
tautological (0 wrong edges / 100% correct edges when compared to itself) and carry no
information. `TV_Confounding` is the one number here that's actually a description of the
benchmark, not a method score: the total-variation distance between the marginal distributions of
every channel pair the graph says is **not** directly connected.

### What we obtained

| Config | TV_Confounding |
|---|---:|
| smoke (linear) | 0.628 |
| smoke_nl (nonlinear) | 0.305 |

### Implication

The linear SCM's non-adjacent channels are noticeably more statistically entangled (TV=0.63) than
the nonlinear SCM's (TV=0.31) at the same `k=5, sparsity=0.2` graph. This is plausible: the linear
VAR's `A @ x` propagation mixes channels through the full lag structure without any bound, while
the nonlinear mechanism's `tanh`-bounded, spectral-norm-capped MLP transitions saturate and damp
cross-channel influence faster. **If a future graph-discovery method is added**, this number is
the right one to report as "how hard is it to distinguish confounding from a real edge on this
benchmark" — a genuinely useful sanity check that the current pipeline correctly separates from
the (currently meaningless) SHD/LagAcc numbers.

---

## 6. Cross-cutting implementation finding: a tolerance mismatch (IVR / CF-faith retroactive check vs. `derive_intervention_t`)

This is the one issue in this report that looks like a genuine implementation bug worth fixing,
not just an interpretation caveat. Traced directly from the data:

**Symptom**: `CftsCels` scores `IVR = 1.00` (100% "retroactive violation") and `cf_faith_*_soft =
0.0` **exactly** (not just `hard=0` — the soft score underflows to zero) on **every single
instance**, on both configs — a suspiciously uniform, extreme result compared to every other
method (`IVR = 0.00` for all of them).

**Root cause** (verified directly on the persisted arrays):

- `derive_intervention_t(x, x_cf, tol=1e-3)` defines `t0` as the first timestep where the
  **per-element max** `|x_cf[t] - x[t]|` exceeds `1e-3`. By this definition, every element before
  `t0` is individually below `1e-3` — small.
- But `CFfaith`'s retroactive check and `axis_c.ivr` both use a **summed** L1 difference over
  *all* `(t, channel)` pairs before `t0`, compared against a much smaller absolute
  threshold (`CFfaith.tol = 1e-4`, `ivr.eps = 1e-4`).
- For a method like CftsCels whose output reconstructs the *entire* trajectory (not just a
  copy-then-edit), every pre-`t0` element carries a small nonzero reconstruction residual
  (measured directly: mean per-element diff ≈ 5×10⁻⁵–10⁻⁴, always below the `1e-3`
  per-element `derive_intervention_t` threshold). Summed over the ~5–17 pre-`t0` timesteps × 5
  channels, that residual totals **0.0008 to 0.011 — 8x to 100x above the `1e-4` retroactive
  threshold** used by `CFfaith`/`ivr`, even though *no single element* would be flagged as
  "changed" by the method that defines `t0` in the first place.
- Every other method (CARLA, CftsWachter, CftsCOMTE, CftsConfeti, CftsCounts) leaves the
  pre-`t0` region an **exact copy** of the input (`sum ≈ 1e-6` or less), so this never triggers
  for them — the mismatch is invisible unless a method reconstructs the whole series.

**Why this matters**: `IVR` and `CFfaith`'s retroactive gate are meant to catch a method that
*actually* changes the past. CftsCels is being penalized for floating-point-scale reconstruction
noise it is required to carry as a numerical artifact of its architecture (a saliency-guided
encode/decode; it never literally copies the input tensor). This makes both metrics
**over-sensitive by design**: a summed threshold across a growing number of `(t, channel)` pairs
gets easier to trip as `T` and `k` grow, purely from accumulated floating-point noise — independent
of `T`/`k`, the *per-element* `derive_intervention_t` threshold does not.

**Recommended fix** (pick one, not both — they're two different philosophies for the same check):
1. Make `CFfaith`'s retroactive check and `axis_c.ivr` use a **max**, not a **sum**, over the
   pre-`t0` region (`np.abs(...).max() > tol` instead of `.sum() > tol`) — this makes the
   retroactive gate consistent with the very definition of `t0` it is downstream of, and removes
   the `T×k`-dependent sensitivity growth entirely.
2. If the sum is intentional (e.g. to catch many *tiny*, individually-invisible-but-collectively-
   real retroactive edits — a real class of violation a max-based check would miss), raise
   `CFfaith.tol`/`ivr.eps` to be consistent with `derive_intervention_t`'s per-element `1e-3`
   scaled by the expected pre-window size, or better, apply the summed check with a `mean` (already
   used for the *forward* residual — `l1_residual = ... .mean()` in `cf_faith.py:185`) rather than
   `sum`, for scale-invariance with `T`.

Either fix should be validated against the **golden linear test** (`tests/test_golden_linear.py`)
before landing — the retroactive check's current sum-based semantics were part of the v0.1-locked
behavior, so this is a considered change, not a drop-in patch.

> **RESOLVED (M1, 2026-07-07) — fix #1 adopted.** The retroactive gate in `CFfaith` and the
> per-element threshold in `axis_c.ivr` now use the **per-element max** predicate at the shared
> constant `causaltemp_xai.scm.intervention.INTERVENTION_TOL = 1e-3` — the same predicate, at the
> same scale, that `derive_intervention_t` uses to *define* `t0`. Why #1 and not #2: (i) internal
> consistency — a region certified "unchanged" by the definition of `t0` can then never be flagged
> retroactive by its own downstream consumer; (ii) `T×k`-invariance — the summed check's
> false-positive rate grew with sequence length from accumulated float noise alone; (iii) fix #2
> (rescaling the sum) keeps a window-size-dependent threshold that must be re-derived per config.
> Residual limitation, accepted deliberately: many individually-sub-tolerance but collectively-real
> retroactive edits are not caught — but that class is exactly what the per-element
> `derive_intervention_t` heuristic already defines as "no change", so catching it belongs to a
> declared-`t0` protocol, not this gate. Validated against the golden tests with **no constant
> re-blessed** (the golden CFs have exactly-zero pre-`t0` regions and the golden retro case is a
> per-element 5.0 violation, so both philosophies agree on every pinned case). Regression coverage:
> `tests/test_metric_adversarial.py::TestCelsFalseFlagRegression`. Consequence for IVR: when
> `T_int` is *derived* by `derive_intervention_t` with the same tolerance (as this pipeline does),
> IVR = 0 by construction — there it is a pipeline-consistency canary; it is discriminative only
> for methods that *declare* their own intervention time (e.g. Phase-05 oracles). Stated in the
> `ivr` docstring.

---

## 7. Mutual exclusivity of CF-faith semantics — clarifying scope

The plan documents (and Stage 4's `test_structural_cf.py`) establish that the **oracle**
structural-CFs are mutually exclusive: the noisy oracle satisfies `pearl_hard=1` and the
noiseless-skeleton oracle satisfies `rollout_hard=1`, never both — this was re-verified directly on
`smoke_nl`'s oracle output (0 violations out of 20 rows).

**For real CF methods, this is not a general guarantee** and does not need to be one. Checked
directly: `CftsConfeti` scores `rollout_hard=1 AND pearl_hard=1` simultaneously on 7/10 (`smoke`)
and 4/10 (`smoke_nl`) instances. This happens because CftsConfeti's edits are tiny relative to the
SCM's noise scale, so both the noiseless-rollout reference and the noise-abducted pearl reference
land within `tol` of the (barely-changed) CF at the same time — a legitimate near-degenerate case,
not a metric bug. **Documentation note**: if "mutual exclusivity" is cited elsewhere as a benchmark
property, scope it explicitly to the oracle construction, not to arbitrary CF outputs.

---

## 8. Summary — what to trust, what to double-check, what to fix

| Finding | Status |
|---|---|
| CARLA / oracle causal faithfulness vs. every non-causal method's near-zero faithfulness | **Trust** — reproduces cleanly on both mechanisms, matches the plan's phenomenon guard |
| CftsConfeti's partial faithfulness via small edits | **RESOLVED 2026-07-07** (was: trust-but-gameable) — joint faithfulness–validity criterion `cf_faith_*_hard_valid` added to eval + tables; tiny-edit CFs earn no faithfulness credit unless they also flip the classifier (§9.4) |
| CftsCOMTE's negative OOD score | **Trust** — plausible (shapelet swap), consistent across configs |
| `shift_vr` ratios computed from small `validity_base` (e.g. CftsConfeti's 2.67x) | **RESOLVED 2026-07-07** — ratio reported only when `validity_base ≥ 0.3`; the `(validity_base, validity_shift)` pair is always reported (§9.3) |
| IG `InputSens` higher on nonlinear (0.13 vs 0.02) | **Plausible, worth re-checking at scale** — could reflect a genuinely less-regularized nonlinear classifier (M2 scope, not a metric defect) |
| Axis A `ICC ≈ 0.20` (chance-level for k=5) | **Correctly implemented; interpretation caveat** — measures alignment with an oracle intervention IG was never asked to explain |
| Axis A `MCC_coverage = 1.00` (ceiling) | **RESOLVED 2026-07-07** — redesigned as chance-normalized continuous causal mass (threshold-free, scale-invariant, 1.0 = chance); ceiling structurally removed (§9.2) |
| Axis B `SHD=0`/`LagAcc=1.0` | **Not a bug, but not informative** — tautological without a graph-discovery method; `TV_Confounding` is the real signal here |
| `CftsCels` `IVR=1.00` / `CF-faith soft=0.0` on every instance | **RESOLVED 2026-07-07** — §6 fix #1 adopted: per-element max retro gate at the shared `INTERVENTION_TOL=1e-3`; CELS scores regenerated, artifact rows purged from `results/tables/` (§9.1) |
| "Mutual exclusivity" of CF-faith semantics | **Correctly scoped to the oracle construction only** — real methods can trip both when edits are small; not a bug, just needs the caveat documented wherever this claim is repeated |

Everything in this report was generated from real pipeline runs (`smoke`/`smoke_nl`, fully-trained
classifiers, `n_cf=10`) and the underlying arrays were re-inspected directly (not just the
aggregate JSON) wherever a number looked suspicious — the two "fix recommended" rows above are
traced to root cause on the actual persisted `X_sel.npy`/`X_cf_*.npy` data, not guessed from
symptoms.

---

## 9. M1 resolution log (2026-07-07) — all "fix recommended" rows closed

Owner: Methodology & Coding Scientist. Every fix below is covered by
`tests/test_metric_adversarial.py` (40 adversarial/regression tests) and validated against the
frozen golden tests (`tests/test_golden_linear.py` — **no constant re-blessed**; the per-element
retro gate agrees with the summed gate on every pinned golden case, so golden semantics are
unchanged). Full suite after all fixes: see the M1 close-out report.

### 9.1 Retroactive-tolerance mismatch (CELS false flag) — P0

* **Fix**: §6 option 1. New shared constant `INTERVENTION_TOL = 1e-3`
  (`causaltemp_xai/scm/intervention.py`); `CFfaith` gained `retro_tol=INTERVENTION_TOL` and its
  retroactive gate is now a **per-element max** (was: summed L1 vs `1e-4`); `axis_c.ivr`'s `eps`
  default is now `INTERVENTION_TOL` (was `1e-4`).
* **Rationale (one line)**: the retro check must use the same per-element predicate at the same
  scale as the `derive_intervention_t` rule that defines `t0`, otherwise it flags as "edited" a
  region its own upstream definition certified as unchanged; full argument in §6's resolution note.
* **Files**: `causaltemp_xai/scm/intervention.py`, `causaltemp_xai/metrics/cf_faith.py`,
  `causaltemp_xai/metrics/axis_c.py`.

### 9.2 `MCC_coverage` ceiling

* **Fix**: `mcc_concept` redesigned as **chance-normalized continuous causal mass** (threshold
  parameter removed); details and why-not-a-relative-threshold in §4's resolution note.
* **Rationale (one line)**: any thresholded coverage of a dense saliency map sits at the 1.0
  ceiling regardless of where the threshold is set — only a continuous, scale-invariant mass
  formulation restores discriminative power.
* **Files**: `causaltemp_xai/metrics/axis_a.py`.

### 9.3 `shift_vr` instability

* **Fix**: ratio gated on `validity_base ≥ MIN_VALIDITY_BASE_FOR_RATIO = 0.3` (else `None`);
  `(validity_base, validity_shift)` pair + `n_base`/`n_shift` always reported; details in §3's
  resolution note.
* **Rationale (one line)**: a retention ratio on ≤3 base successes at `n_cf=10` moves by integer
  multiples per instance — below the floor the pair is the honest report and the ratio is noise.
* **Files**: `causaltemp_xai/eval.py`, `tests/test_shift.py` (contract updated).

### 9.4 Gameability loophole (tiny-edit "faithfulness")

* **Fix**: joint criterion `cf_faith_{rollout,pearl}_hard_valid` = per-instance
  `hard × 1[valid]`, wired through `evaluate_method`, `per_instance_records`,
  `aggregate_method_row`, the summary console table, and
  `results/tables/table_axis_c_cf_faith.csv`; details in §1's resolution note.
* **Rationale (one line)**: faithfulness credit is only meaningful for a counterfactual that
  actually flips the classifier; requiring validity jointly (not proximity — small edits are a
  virtue when valid) closes the loophole without penalising genuinely good sparse CFs.
* **Files**: `causaltemp_xai/eval.py`, `experiments/_common.py`.

### 9.5 Incidental defect found during the M1 re-run

* `causaltemp_xai/methods/counterfactual/cfts_methods.py` registered the vendored cfts submodule
  via `Path(__file__).parents[2]` — stale after the restructure moved the file one directory
  deeper (correct: `parents[3]`). It silently worked only while the now-deleted duplicate
  `methods/cfts_methods.py` (correct depth) registered the path first; after M0's cleanup every
  cfts-backed method failed with `ModuleNotFoundError: cfts`. Fixed 2026-07-07.

### 9.6 Post-fix empirical verification (regenerated `smoke`/`smoke_nl`, `n_cf=20`)

Full phased pipeline (`experiments/01`–`06`) re-run end-to-end on 2026-07-07 with the M1 fixes in
place (`smoke`: LSTM, 92% test acc, `n_cf=20`; `smoke_nl`: oracle-only path per the documented
nonlinear routing — no classifier/CF-method phase there, so shift_vr/Axis-A below are `smoke` only).

**CftsCels — the flagship fix, before vs. after** (`results/tables/table_axis_c_cf_faith.csv`):

| | IVR | rollout_hard | rollout_soft | pearl_hard | pearl_soft |
|---|---:|---:|---:|---:|---:|
| Before (audited, `n_cf=10`) | **1.00** | 0.00 | **0.00** | 0.00 | **0.00** |
| After (regenerated, `n_cf=20`) | **0.00** | 0.00 | **0.784** | 0.00 | **0.982** |

`IVR` drops from the false-positive `1.00` to `0.00` (no genuine retroactive edit — confirmed
directly, CftsCels never touches the pre-`t0` region by more than per-element `1e-3`), and the soft
CF-faith scores — previously hard-zeroed by the same false retroactive gate before any forward
comparison was even attempted — now surface CftsCels's true (non-causal, but not retroactive)
continuation residual: `0.784` under noiseless rollout, `0.982` under Pearl. `hard` stays `0` on
both semantics, correctly — CftsCels was never claimed to be an exactly-faithful method, only
wrongly flagged as *retroactive*.

**`MCC_coverage` — ceiling removed** (`results/smoke/lstm/axis_a_attribution.json`): `1.00` (every
config, every run, pre-fix) → **`1.278`** post-fix (`smoke`, `n=20` scored instances) — a
chance-normalized value now free to move (see §4/§9.2); `1.278` reads as "≈28% more attribution
mass on the true causal parents than a uniform map would place there," a modest but non-trivial,
non-ceilinged signal. `ICC = 0.1998` (unchanged in form, near-chance as expected — §4).

**`shift_vr` — CftsConfeti's swing resolved by scale, guard verified live**
(`results/smoke/lstm/shift_vr.json`): at `n_cf=20`, `validity_base=0.55` for CftsConfeti — already
above the `0.3` floor, so the ratio is reported (not suppressed) and is now a sane `1.09`
(`validity_shift=0.60`), not the `2.67` small-sample artifact from `n_cf=10`. Every method's
`(validity_base, validity_shift, n_base, n_shift)` is present in the JSON regardless, confirming
the "pair always reported" half of the fix independent of the floor.

**Incidental fix (§9.5) confirmed load-bearing**: without the `cfts` submodule path-depth
correction, phase 03 fails outright for every cfts-backed method (`ModuleNotFoundError: cfts`) —
this was hit and fixed during this same re-run, not a hypothetical.

**Artifact provenance**: `results/tables/table_axis_c_cf_faith.csv` (smoke + smoke_nl-oracle rows),
`results/smoke/lstm/{summary.json,per_instance.csv,axis_a_attribution.json,shift_vr.json}`,
`results/smoke_nl/oracle/summary.json`, `results/figures/fig{1,2,3}_*.png` — all regenerated
2026-07-07 from this fixed codebase; no number in this section is carried over from the pre-fix
audit.
