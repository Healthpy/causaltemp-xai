# M2 — Multi-seed/Bootstrap-CI Infrastructure + Pearl-CARLA (Task #8, #10)

**Date**: 2026-07-08 · **Owner**: Methodology & Coding Scientist
**Scope**: this document covers *only* tasks #8 (multi-seed + bootstrap-CI aggregation) and
#10 (Pearl-CARLA noise-reinjecting recourse variant) from the M2 plan. Tasks #9/#11
(Transformer classifier, LSTM-vs-Transformer generalization table) were **cancelled by
PI decision mid-task** (classifier landscape for M2 is LSTM-only) and are not covered here;
any Transformer-classifier files that were briefly created were discarded and are not part
of this branch.

This is a **new, additive** document — it does not edit `docs/axis_metrics_report.md`
(M1, frozen except for its own §9 resolution log) or `docs/hypotheses_assessment.md`
(v0.1, frozen).

---

## 1. Multi-seed + bootstrap-CI aggregation (Task #8)

### 1.1 Design

**Problem**: every stage of the pipeline (SCM generation, train/val/test split, LSTM
training, CF-instance selection) is deterministic given one `BenchmarkConfig.seed`. A
single seed's numbers cannot support a confidence interval — there is nothing to resample
across except individual CF instances within that one run, which understates the true
uncertainty (a different seed draws a different SCM graph/mechanism, a different LSTM
fit, and a different flip-candidate ordering).

**Fix, three pieces:**

1. **`causaltemp_xai.config.seeded_variant(base, seed, name=None)`** — mirrors the
   existing `shifted_config` pattern (vary exactly one field of a frozen
   `BenchmarkConfig`, everything else identical). Changing only `.seed` re-seeds every
   downstream stage in one call, because `data_io.build_generator`, `stratified_split`,
   and `LSTMClassifier.__init__` all key off `cfg.seed` already. The returned config's
   name defaults to `f"{base.name}_seed{seed}"` — a fresh, unambiguous on-disk directory
   even when `seed` happens to equal `base.seed` (deliberately not aliased to the
   original un-suffixed run, so every seed replicate — including a literal seed=0 rerun —
   is traceable to its own directory).

2. **`--seed` added to phases 01-04** (`experiments/01_generate_benchmarks.py` through
   `04_evaluate_axes.py`). Each phase's core function (`generate_one`/`train_one`/`run`)
   gained an optional `seed: int | None = None` parameter; when set, it calls
   `get_config(name)` then `seeded_variant(cfg, seed)` before doing anything else, so the
   phase transparently operates on the seeded config's name throughout (dataset
   directory, checkpoint path, results directory, table rows). Default `None` is
   byte-identical to the pre-existing behavior (verified: full suite green before and
   after this change, and a `--seed`-less invocation touches no new code path since the
   `if seed is not None` guard is a no-op).

3. **`experiments/07_aggregate_seeds.py`** (new phase) — orchestrates phases 01→04 once
   per requested seed (via direct function calls through `importlib.import_module`, not
   subprocess, since numbered-prefix filenames are importable — verified — but not valid
   `import` statement targets), then pools every method's per-instance metrics across
   seeds and reports a **hierarchical (seed-cluster) bootstrap 95% CI** on every mean.

### 1.2 Why a hierarchical bootstrap, not a flat one (`causaltemp_xai/stats.py`)

A flat percentile bootstrap that pools every instance from every seed into one i.i.d.
sample would **understate** the reported uncertainty whenever seeds differ
systematically — e.g. one seed's LSTM fit is a few points worse, or one seed's SCM draw
happens to be easier for a given CF method. That between-seed variability is exactly
what the multi-seed protocol exists to quantify (plan Standing Decision #4: "every
headline number carries a CI from M2 onward").

`causaltemp_xai.stats.hierarchical_bootstrap_ci` therefore implements a two-level cluster
bootstrap: for each of `n_boot` iterations, (1) resample the *seeds* with replacement,
(2) resample the *instances within each chosen seed's run* with replacement, (3) pool and
recompute the statistic. `causaltemp_xai.stats.bootstrap_ci` (flat, single-level) is kept
alongside it for the genuinely-i.i.d. case (e.g. per-instance values within one seed).

**Why not `scipy.stats.bootstrap`**: it has no first-class two-level cluster-resampling
mode — it is built for a flat i.i.d. sample (exactly what `bootstrap_ci` covers), not the
(seed, instance) hierarchy the multi-seed protocol needs. A hand-rolled implementation
was the more transparent and directly-correct option for this specific structure.

Validated (`tests/test_stats.py`, 18 tests): determinism given a fixed resampling seed;
`ci_lo <= mean <= ci_hi`; narrower CIs for lower-variance data; correct NaN/empty/n=1
edge-case handling (degenerate point CI, never a fabricated spread); and — the key
property for the hierarchical estimator — a synthetic check that seeds with wildly
different (but internally constant) per-seed means produce a **wider** CI than
homogeneous seeds with the same total sample size, confirming the between-seed
variability is actually captured, not averaged away.

### 1.3 Aggregation output (`experiments/_common.py::aggregate_across_seeds`)

Reads `results/<config>_seed<seed>/lstm/per_instance.csv` for every requested seed,
groups by `method`, and for each metric in `SEED_AGGREGATE_METRICS` (`validity`,
`proximity_l1`, `proximity_l2`, `sparsity`, `trsi`, `ivr`, both CF-faith semantics'
hard/soft scores, and the M1 joint faithfulness-validity columns) computes
`hierarchical_bootstrap_ci` over the seed-grouped values. Output is one row per method,
wide format: `<metric>_mean`, `<metric>_ci_lo`, `<metric>_ci_hi`, `<metric>_n` for every
metric, written to `results/tables/table_seed_aggregate_<config>_lstm.csv`.

### 1.4 Smoke-scale validation (real pipeline run, not synthetic data)

Ran `experiments/07_aggregate_seeds.py --config smoke --seeds 0 1 2 --n-cf 10 --methods
CARLA CftsWachter CftsCOMTE --n-boot 3000` end-to-end (phases 01→04 for each of 3 seeds,
then aggregation). Scope was deliberately reduced from a full 6-method run: a timing
probe (`n_cf=3`, 3 methods) measured ~108s, dominated by Shift-VR's CF regeneration; the
full 6-method, `n_cf=20` CF-generation phase took ~50 minutes *per seed* at smoke scale in
the M1 close-out, so a 3-seed × 6-method run was not attempted synchronously in this
session per the explicit scope limit (do not launch/wait on long runs). The reduced-scope
run (3 seeds × 10 instances × 3 fast methods) completed in a few minutes and is a genuine
run through the real pipeline, not a synthetic/mocked test of the aggregation code alone.

**Per-seed results** (from the run's console log, `results/smoke_seed{0,1,2}/lstm/`):

| Seed | test_acc | CARLA validity | CftsCOMTE validity | CftsWachter validity |
|---|---:|---:|---:|---:|
| 0 | 0.940 | 1.00 (10/10) | 1.00 (10/10) | 1.00 (10/10) |
| 1 | 0.880 | 1.00 (10/10) | 1.00 (10/10) | 1.00 (10/10) |
| 2 | 0.970 | 1.00 (10/10) | 1.00 (10/10) | 0.80 (8/10) |

**Pooled aggregate table** (`results/tables/table_seed_aggregate_smoke_lstm.csv`,
`n_boot=3000`, `n=30` pooled instances per method):

| Method | validity mean [95% CI] | cf_faith_rollout_hard [CI] | cf_faith_pearl_hard [CI] |
|---|---|---|---|
| CARLA | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| CftsCOMTE | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CftsWachter | 0.933 [0.767, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

Unit coverage: `tests/test_config.py` (7 tests: `seeded_variant` changes only `seed`,
gets a distinct name even when `seed == base.seed`, is deterministic, and — the property
that actually matters for the protocol — two different seeds produce a genuinely
different graph/mechanism, not just different noise). This is independently corroborated
by the real run above: `TV_Confounding` (a graph-structure statistic, phase 01's console
output) differed across all three seeds (0.627, 0.277, 0.566), confirming
`seeded_variant` does vary the SCM itself, not only the noise draw.

This is exactly the validation signature the task asked for: **CARLA and CftsCOMTE**
(validity = 1.0 in every instance of every seed) get a **degenerate, razor-narrow CI**
(a bootstrap over a constant sample cannot produce spread — correct behavior, not a bug).
**CftsWachter** (1.00, 1.00, 0.80 across the three seeds — one genuinely worse seed)
gets a **visibly wider CI**, `[0.767, 1.000]`, correctly propagating the one bad seed's
signal into the pooled uncertainty rather than averaging it away. `proximity_l1` (a
continuous, more variable metric) shows the same pattern at a finer grain, e.g. CARLA's
`proximity_l1 = 35.5 [11.3, 61.3]`, reflecting the real per-seed means of 31.4, 11.1, 64.0
— a wide, honest CI on a metric with real cross-seed variance. Full CSV:
`results/tables/table_seed_aggregate_smoke_lstm.csv` (all 12 metrics × mean/ci_lo/ci_hi/n).

---

## 2. Pearl-CARLA — noise-reinjecting recourse variant (Task #10)

### 2.1 Design

`causaltemp_xai/methods/counterfactual/carla.py` gains `PearlCARLARecourse`, added
**alongside** `CARLARecourse` (not a subclass, not a `semantics=` flag on the existing
class) — see the class docstring for the full rationale; in short: `CARLARecourse`'s
current behavior is pinned by `tests/test_methods.py::TestCARLA` and the M1-regenerated
`results/` `rollout_hard=1` rows, so it is left **byte-for-byte untouched**, and
`PearlCARLARecourse` duplicates the small optimization-loop structure rather than
refactoring a shared base out from under `CARLARecourse` — a deliberate low-risk choice
given the hard constraint not to perturb `CARLARecourse`'s pinned numbers, and consistent
with this codebase's existing convention of small per-class duplication over
shared-base abstraction (`cfts_methods.py`'s six near-identical `generate_batch`s).

Mechanically: `PearlCARLARecourse` abducts the exogenous noise from the **factual**
trajectory once per `generate()` call — `eps[t] = x_orig[t] - mechanism.forward_numpy(window)`
for every `t >= 1` (exact under this benchmark's additive-noise mechanism family) — the
identical construction `CFfaith(semantics="pearl_delta")` uses internally to build its
own reference trajectory. It then rolls the recourse forward reinjecting that noise,
`x_cf[t] = mechanism.forward_torch(window) + eps[t]` for `t > t0`, instead of
`CARLARecourse`'s noiseless `x_cf[t] = mechanism.forward_torch(window)`. Everything else
(actionability masking, t0-candidate search, Adam optimization of `delta` at `t0`,
flip-then-proximity tie-break across candidates) is identical to `CARLARecourse`.

### 2.2 Invariant that holds "by construction"

Not zero-retroactive-change-is-different, as the task brief speculated — that invariant
holds **identically** for both classes (both copy `x[:t0]` verbatim into the CF
regardless of forward-rollout semantics; verified empirically, `atol=1e-6`). The actual
distinguishing invariant is which `CFfaith` semantics scores `hard=1`:
`CARLARecourse` → `cf_faith_rollout_hard=1`; `PearlCARLARecourse` → `cf_faith_pearl_hard=1`
— confirmed at smoke scale across every tested horizon (`pearl_hard=1.00` in every
condition tested, including the ones where validity was low — faithfulness and validity
are measured independently, and construction only guarantees the former).

### 2.3 Empirical finding — the long-horizon story is more interesting than "Pearl fixes it"

The task brief (and the v0.1 write-up it draws from) frames Pearl-CARLA as expected to
"recover validity while `pearl_hard=1`," implicitly assuming noise-reinjection is
strictly better at long horizons than `CARLARecourse`'s noiseless rollout. **Smoke-scale
measurement (linear SCM, `k=5, T=30, N=300, seed=0`, 15 flip-candidate instances)
contradicts the naive version of that expectation**:

| t0_frac (horizon) | CARLA validity (default `lam_prox=0.5`) | PearlCARLA validity at CARLA's `lam_prox=0.5` | PearlCARLA validity at `lam_prox=0.1` (new default) |
|---|---:|---:|---:|
| 0.1 (horizon 27/30) | 0.40 | **0.07** | 0.20 |
| 0.3 (horizon 21/30) | 1.00 | — | 0.93 |
| 0.5 (horizon 15/30) | 1.00 | — | 1.00 |
| 0.7 (horizon 9/30)  | 1.00 | — | 1.00 |

At CARLA's own default `lam_prox=0.5`, Pearl-CARLA's validity at the longest tested
horizon is *worse* than CARLA's (0.07 vs 0.40) — reinjecting noise made the collapse
**worse**, not better. Root cause, confirmed both mathematically and by a hyperparameter
sweep (`lam_prox` 0.5 → 0.05 → 0.0 at fixed horizon: validity 0.07 → 0.47 → 1.00): the
Pearl delta obeys the *homogeneous* recursion `delta[t] = sum_l A_l @ delta[t-l]` (the
factual noise cancels exactly — see `CFfaith`'s `pearl_delta` docstring), and this
benchmark's SCM stability requirement (spectral radius < 1, empirically 0.9 on this
config) means any one-shot intervention's effect decays geometrically. Surviving a long
horizon therefore requires a **proportionally larger** `delta[t0]` than CARLA's noiseless
variant needs — and CARLA's default `lam_prox=0.5` quadratically over-penalizes exactly
that larger delta.

This is not a bug in `PearlCARLARecourse` — reinjecting the true, already-realized noise
means the counterfactual becomes progressively harder to distinguish from the *factual*
trajectory (which, by construction, is *not* the target class) as the intervention's
effect fades. This is arguably a more scientifically honest result than CARLA's own
long-horizon behavior: CARLA's noiseless trajectory converges toward a
class-*independent* fixed point regardless of the intervention (a property of the
homogeneous linear map having no reference to noise at all), so its apparent long-horizon
"robustness" may itself be an artifact of converging to a universal attractor that
happens to fall on the target-class side of the decision boundary, not evidence of a
successfully-optimized recourse.

**Resolution adopted**: `PearlCARLARecourse`'s default `lam_prox` is set to **0.1** (not
`CARLARecourse`'s 0.5) — a deliberate, documented, empirically-motivated choice specific
to the new class (no pinned test/results depend on its defaults, unlike
`CARLARecourse`). At this default, full `n_steps=500` (CARLA's own default, not the
reduced value used only for the diagnostic hyperparameter sweep), Pearl-CARLA matches
CARLA's validity at 3 of the 4 tested smoke-scale horizons and substantially narrows the
gap at the most extreme one (0.20 vs CARLA's 0.40), while `pearl_hard=1.00` holds
unconditionally throughout (a property of the rollout construction, independent of
`lam_prox`).

**What this needs from the PI**: a decision on whether `lam_prox=0.1` is an acceptable
permanent default, or whether a smarter fix (e.g. scaling `lam_prox` by an estimate of
the SCM's spectral radius, or a per-instance line-search over `lam_prox`) should be
pursued before `full`/`full_nl` (`T=100`, much longer horizons, where this effect will be
far more pronounced — `0.9^70 ≈ 6×10⁻⁴` vs `0.9^27 ≈ 0.058` at smoke scale, i.e. the
same `lam_prox` value will very likely need to be lower still, or the delta bound raised,
for the full-scale horizon). This is exactly the kind of finding that should gate the
full-scale run's hyperparameters rather than be discovered only after a multi-hour run
completes.

### 2.4 Files touched

* `causaltemp_xai/methods/counterfactual/carla.py` — new `PearlCARLARecourse` class,
  `CARLARecourse` untouched.
* `causaltemp_xai/methods/counterfactual/__init__.py`,
  `causaltemp_xai/methods/__init__.py` — export `PearlCARLARecourse`.
* `tests/test_methods.py` — new `TestPearlCARLA` class (shape/finite, zero retroactive
  change, `cf_faith_pearl_hard==1` by construction, batch-shape).
