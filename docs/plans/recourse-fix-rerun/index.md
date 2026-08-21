# Plan: Recourse and metric fixes for the second Helios run

**Date**: 2026-08-21
**Branch**: `run-full-experiments`
**Predecessors**: `docs/full_run_2026-08-18.md` (run 1 report), `docs/general_plan.md`
**Goal**: Fix the CARLA/PearlCARLA delta-collapse, make the shipped metric tables
self-consistent, harden the Helios harness, and rerun `full` + `full_nl` on Helios GH200.

---

## Context

Run 1 completed on Helios on 2026-08-18 (jobs 20797977/78/79, ~26 GPU-hours, all exit `0:0`).
It produced `results_helios/tables/final_table_long.csv` (1251 rows, 5 datasets x 7 methods
plus 2 oracle controls) and the report `docs/full_run_2026-08-18.md`. A four-agent audit on
2026-08-21 established that the table cannot support a method ranking.

### The primary defect: one bug, not two

`docs/full_run_2026-08-18.md` section 7.1 reports two separate failures on `full_nl`:
PearlCARLA fully degenerate (`frac_degenerate=1.00`, `n_cf_faith_scorable=0`,
`proximity_l1~=7.5e-05`) and CARLA fully vacuous (`validity=0.000`, `frac_vacuous=1.00`,
`proximity_l1=111.6`, `do_complexity=74`). They are the same failure seen through two
different rollout semantics.

Both classes minimise `lam_pred * CE + lam_prox * ||delta||^2`
(`causaltemp_xai/methods/counterfactual/carla.py:288` and `:511`). That objective has a
closed-form stationary point at `delta* = -(lam_pred / (2*lam_prox)) * dCE/ddelta`.
Instrumenting the real loop on regenerated `smoke_nl` reproduces it to 4 significant figures:

| method | `lam_prox` | `dCE/ddelta` at 0 | converged `delta` | predicted `grad/(2*lam_prox)` |
|---|---|---|---|---|
| CARLA | 0.5 | 1.0641593e-05 | 1.0641516e-05 | 1.0641593e-05 |
| PearlCARLA | 0.1 | 3.1715e-05 | 1.5855e-04 | 1.5858e-04 |

Gradient norm at step 499 is ~1e-12. The optimiser converges cleanly to the true global
optimum of its own objective. It is **not** a step budget, a swallowed exception, a detached
tensor, tanh zeroing the gradient, or a failed abduction.

`dCE/ddelta` is ~1e-5 because the intervention cannot reach the readout. The LSTM classifies
from the last hidden state only (`causaltemp_xai/classifiers/lstm.py:110`,
`last_hidden = hn[-1]`); measured `dL/dx_cf` is 2.7e-06 at the intervention row against
3.76e-04 at the final row. The retuned MLP mechanism then contracts at 0.15-0.33 per step
over `T - t0` steps (22 on `smoke_nl`, 75 on `full_nl`).

Every nonlinear delta lands 1-5 orders of magnitude below `INTERVENTION_TOL = 1e-3`
(`causaltemp_xai/scm/intervention.py:36`):

| config | method | delta at `t0` | derived `intervention_t` | reported symptom |
|---|---|---|---|---|
| `full_nl` | CARLA | 6.4e-08 | 26 (= t0+1) | `frac_vacuous=1.00` |
| `full_nl` | PearlCARLA | 7.5e-08 | 99 (= T-1) | `frac_degenerate=1.00` |
| `smoke_nl` | CARLA | 3.3e-05 | 9 (= t0+1) | `frac_vacuous=1.00` |
| `smoke_nl` | PearlCARLA | 1.7e-04 | 29 (= T-1) | `frac_degenerate=1.00` |
| `full` (linear) | CARLA | 5.4e-03 | 25 (= t0) | "works", `validity=1.000` |
| `full` (linear) | PearlCARLA | 2.9e-02 | 25 (= t0) | `validity=0.000` (section 7.4) |

The decision boundary is reachable. A brute-force sweep over delta magnitude, channel and
sign on regenerated `smoke_nl` flipped 20/20 candidates at both `t0=8` and `t0=15`, under both
semantics, at a median `||delta||` of 20. But `lam_prox * ||delta||^2` at `||delta||=20` is
`0.5 * 400 = 200`, against a maximum achievable CE reduction of 3-5. The penalty outweighs the
reward by 40-70x. The objective is correctly *formed*; its `lam_prox` is mis-scaled by roughly
two orders of magnitude for this regime.

### The consequence that invalidates run 1's linear tier

Setting `delta` to **exactly zero** — no optimiser at all — and measuring validity:

```
full (linear)  t0=25:  noiseless 100/100 flip    pearl 0/100
full (linear)  t0=50:  noiseless  92/100 flip    pearl 0/100
smoke_nl (mlp) t0=7 :  noiseless   0/20  flip    pearl 0/20
```

CARLA's `validity = 1.000` on the linear `full` tier is **100% noise-deletion** by the
noiseless rollout (`carla.py:200-225`), requiring no intervention whatsoever. It escapes the
vacuity flag only because its delta (5.4e-3) happens to land 5x above the 1e-3 tolerance.
The nonlinear mechanism did not break CARLA; it removed the free pass that was concealing the
identical delta-collapse on linear data.

Therefore `docs/full_run_2026-08-18.md` section 7.2 (linear tier as a working baseline) does
not hold, and section 7.4 ("PearlCARLA validity collapses at full scale") is not a separate
horizon phenomenon but the same bug, visible on linear because Pearl gets no free pass.

### Prior art in-repo

This exact failure was already diagnosed and left unfixed. `tests/test_metric_adversarial.py`
lines 404-410 records it, dated 2026-07-31, cross-referencing RISK-17. CI
(`.github/workflows/ci.yml:70-78`) runs the phased pipeline on `smoke` only, never `smoke_nl`,
so no automated check exercises the nonlinear path.

### Six independent defects in the metric and reporting layer

| ID | Defect | Evidence | Impact |
|---|---|---|---|
| D1 | Axis A compares the ground-truth graph **to itself**. The result files say so in their own `note` field. The real graph-corruption + `dynotears` sweep exists in the archive but was never run on Helios; `results_helios/` has no `table_graph_quality_*.csv`. | `results_helios/*/axis_a_benchmark.json` | Section 7.2.3 is not a finding |
| D2 | Validity is **bimodal** across seeds. Median seed-SD 0.110, p75 0.261, p90 0.520; 18/66 archived cells have range >= 0.5, 4 cells are cleanly bimodal (every seed at 0.0 or 1.0). Two Helios point estimates fall outside the archived 3-seed 95% CI. Seed variance exceeds within-seed binomial SE (0.050 at n_cf=100) by 2-5x. | `results/*_seed{0,1,2}/lstm/summary.json` | Every ranking rests on one draw |
| D3 | `cf_faith_*` is a semantics-identity detector, not a quality score. `cf_faith_rollout_hard` takes exactly 2 distinct values across all 33 populated cells and equals 1.0 **iff** the method is CARLA or OracleCF-Rollout, on all 5 datasets. The soft variant ties the oracle to within 4.6e-07. | `results_helios/tables/final_table_long.csv` | Cannot rank within a semantics |
| D4 | `do_complexity_mean` in `summary.json` averages Pearl-semantic schedule lengths over all instances, while `pns.json` averages only over instances with a non-empty Pearl schedule. Both ship under near-identical names. The earlier claim that the denominator difference is exactly `1 - frac_vacuous` is false for `full_nl/CARLA`: both means are 74 while noiseless-semantic `frac_vacuous=1.0`. | `summary.json` vs `pns.json` | Corrupts the section 7.2.1 headline and conflates two semantic predicates |
| D5 | `cf_faith_*_hard_valid` is correctly implemented as the joint rate `P(hard-faithful AND valid)`, so `0.0` when no CF is valid is not a defect. The missing statistic is the separately named conditional rate `P(hard-faithful | valid)`, which must be NaN when there are no valid CFs. | `causaltemp_xai/eval.py`, `experiments/_common.py` | A joint rate cannot answer "faithful when valid" |
| D8 | Oracles emit no `validity`, `do_complexity`, `ood`, `shift_vr` or PNS — 140/350 = 40% coverage. `OracleCF-Pearl` performs exactly one `do()` at `t0=T//2` on one node (`experiments/_common.py:613-632`, `ORACLE_SHIFT = 1.5`), so its do-complexity is **1.0 by construction**: the natural calibration anchor for the axis carrying the paper's headline, and it is absent (T3 shows `n/a`). | `results_helios/tables/final_table_wide.csv` | No calibration anchor |

Also recorded, not in this plan's scope (see Backlog): D6 (PNS absent for `smoke_nl` and
`smoke_spring`; the `B_model_oracle` vs `C_world_oracle` contrast is exactly equal in 19/20
cells so `delta_outcome` carries no signal), D7 (Axis B suppression via
`MIN_VALIDITY_BASE_FOR_RATIO = 0.3` at `causaltemp_xai/eval.py:259` is systematically
correlated with the quantity being studied; the code's own docstring at `eval.py:326-331` says
multi-seed CIs are required to control the regeneration noise), D9 (`scm_noise_plausibility`
has no calibrated ceiling — `OracleCF-Pearl` measures 0.87-0.98, never 1.0, and is beaten by
PearlCARLA and CftsCels on `full`), D11 (`table_axis_c_cf_faith.csv` carries 18 duplicate rows,
the whole `full` block twice at n=3 and n=100), D13 (8 of 13 registered configs never ran on
Helios, including both real-data configs and the RISK-19 interior-label ablation), D16 (the
label is a median threshold on `x[T-1, 0]` and the LSTM head reads `hn[-1]`, so ~1.0 accuracy
is structural and `validity` reduces to "did you move `x[T-1,0]` across the median").

Trustworthy fraction of the intended 1225-cell space: **~70% populated with a defensible
number, under half supporting a ranking claim.**

**Reduced scope (2026-08-21):** the run-1 errata/repository-hygiene stage and the separate
full-scale reachability gate remain removed. The CARLA/PearlCARLA implementation stage has been
restored and relies on the already-recorded regenerated `smoke_nl` reachability evidence.

### Harness state

- The run-1 recipe exists **only as prose** in `docs/full_run_2026-08-18.md` section 3. No
  resubmittable sbatch file reproduces it.
- No script under `slurm/` sets `OMP_NUM_THREADS`. `module load ML-bundle/25.10` pins it to 1,
  and GNU `nproc` obeys that variable, so `export OMP_NUM_THREADS=$(nproc)` is self-defeating.
  Resubmitting `slurm/helios_smoke.sbatch` today runs single-threaded.
- `slurm/full_pipeline_rerun.sbatch` is tracked but targets a different cluster
  (`tesr123566` / `gpu_a100`) and drives the pipeline through `uv run` — the exact pattern
  Helios must avoid, because the committed `uv.lock` resolves torch 2.12.0 against CUDA 13
  while Helios needs torch 2.9.0+cu129 from the site wheelhouse. Silent CPU fallback, not an
  error.
- `experiments/make_final_table.py` defaults to `--results-dir results`, which holds the
  **pre-Helios** numbers. Run 1's output sits in the parallel untracked `results_helios/`.
  Anyone running the script today silently gets the old table.
- `results_helios/` holds 7.2 MB of `.npy` arrays not covered by `.gitignore`'s
  `results/**/*.npy` pattern. A naive `git add -A` would commit them.
- The whole failure reproduces on `smoke_nl` in about 2 minutes on a laptop, matching the
  Helios classifier accuracies exactly (train 0.96 / val 0.98 / test 0.96). Fixes can be
  validated with zero cluster time.

---

## Strategy

Three phases. Nothing touches the cluster until the recourse and metric changes are proven locally.

**Phase A — fix recourse and metric semantics (stages 1-3).** Repair CARLA/PearlCARLA candidate
selection and proximity scaling, expose failed searches separately from classifier validity,
disambiguate do-complexity, add conditional faithfulness, expand oracle coverage, and lock all
contracts with regression tests.

**Phase B — make the run reproducible and validate locally (stages 4-5).** Turn the prose recipe
into committed, `--test-only`-validated sbatch files carrying threading, provenance, and failure
gates. Run the smoke tier end to end before spending grant hours.

**Phase C — run and report (stages 6-7).** Submit the chained Helios jobs, collect, rebuild the
table, verify it against the defect checklist, and document the corrected recourse and metric
semantics.

**Rerun shape (decided 2026-08-21):** repeat the run-1 shape — `full` + `full_nl` at
`n_cf=100`, single seed, ~26 GPU-hours. This validates the fixes at full scale for minimum cost.
It does **not** address D2: the resulting table still has no error bars and no
defence against the bimodality. That is an accepted, documented limitation of run 2, and the
multi-seed campaign is recorded in the Backlog as the natural successor plan.

---

## Success Criteria

| Metric | Baseline (run 1) | Target (run 2) | Rationale |
|---|---|---|---|
| CARLA intervention delta, `full_nl` | 6.4e-08 | > `INTERVENTION_TOL` on >= 90% | Below tolerance there is no action |
| CARLA `frac_vacuous`, `full_nl` | 1.00 | <= 0.20 | Reject the noiseless-continuation free pass |
| PearlCARLA `frac_degenerate`, `full_nl` | 1.00 | <= 0.10 | Restore CF-faith scorable output |
| CARLA classifier `validity`, `full_nl` | 0.000 | > 0.30 | Classifier outcome remains the standard validity metric |
| CARLA `frac_no_cf_found`, `full_nl` | absent | <= 0.20 | Make optimizer failure explicit without redefining validity |
| `do_complexity` naming collision | all-instance and Pearl-scorable values ship under near-identical names | `mean_all` + `mean_pearl_scorable`, with `n_do_scorable` and no claim tying them to `frac_vacuous` | D4 |
| Conditional faithfulness on empty valid set | no separately named conditional metric | `*_hard_given_valid` is NaN and suppressed; existing `*_hard_valid` joint metric remains 0.0 | D5 |
| Oracle metric coverage | 140/350 = 40% | raw `validity`, both explicit do-complexity means and denominator diagnostics present for both oracles; `OracleCF-Pearl` Pearl-scorable mean == 1.0 | D8 |
| Reproducibility | recipe is prose only | committed sbatch, `sbatch --test-only` clean | Run 3 must be one command |
| Publication scope | table auto-discovers retained summaries | final table contains exactly `full` and `full_nl` from run 2 | Prevent stale smoke/run-1 mixing |
| Provenance | run 1 reports `git_dirty:true` | source-scoped `git_dirty:false`, unfiltered state recorded separately | Distinguish code dirt from generated results |

---

## Files That May Be Changed

### Recourse layer (stages 1 and 3)
- `causaltemp_xai/methods/counterfactual/carla.py` -- candidate tie-break, bounded proximity
  backoff, compatibility-preserving status API, and documentation.
- `causaltemp_xai/config.py` -- optional explicit recourse defaults, only if needed.
- `experiments/03_run_cf_methods.py`, `experiments/04_evaluate_axes.py` -- persist and validate
  no-CF sidecars while preserving classifier-only validity.
- `tests/test_methods.py` -- nonlinear recourse and legacy-API regressions.

### Metric and reporting layer (stages 2-3)
- `causaltemp_xai/eval.py` -- `evaluate_method()` conditional-metric NaN handling (D5),
  do-complexity dilution (D4).
- `experiments/_common.py` -- preserve classifier validity and the joint metric; add conditional
  metric aggregation and source-clean provenance.
- `experiments/05_run_oracle_control.py` -- emit the missing oracle metrics (D8).
- `experiments/make_final_table.py` -- preserve joint metrics, add conditional-metric
  suppression provenance, distinguish the two Pearl-semantic do-complexity denominators, and
  require explicit publication configs.
- `tests/test_eval.py`, `tests/test_do_complexity.py` -- cover the new behaviour.
- `tests/test_metric_adversarial.py` -- preserve the joint metric contract and cover the new
  conditional metric semantics.

### Harness (stage 4)
- `slurm/helios_full_run.sbatch` (new), `slurm/helios_smoke.sbatch`,
  `slurm/helios_reports.sbatch` (new), `slurm/helios_setup_env.sbatch`,
  `slurm/helios_import_check.sbatch` -- commit + threading fix and accumulated failure status.

### Documentation (stage 7)
- `docs/full_run_2026-08-2X.md` (new) -- the superseding report.

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Fix the recourse layer](stages/01-fix-recourse-layer.md) | PENDING | | |
| 2 | [Metric reporting fixes](stages/02-metric-reporting-fixes.md) | PENDING | | |
| 3 | [Regression tests](stages/03-regression-tests.md) | PENDING | | |
| 4 | [Helios job scripts](stages/04-helios-job-scripts.md) | PENDING | | |
| 5 | [Local full validation](stages/05-local-full-validation.md) | PENDING | | |
| 6 | [Cluster preflight and submit](stages/06-cluster-submit.md) | PENDING | **HUMAN GATE** | |
| 7 | [Collect, table, document](stages/07-collect-table-document.md) | PENDING | | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

---

## Execution Protocol

This plan is built for **autonomous, unattended execution**, with one explicit exception
noted below. The guiding principle is **keep making progress**: resolve problems in place when
you can, defer them when you can't, and never halt the whole plan over a single fixable or
deferrable issue.

For each stage:

1. **Read the progress tracker** above and pick the stage to work on. If a stage is
   **IN_PROGRESS**, a previous run was interrupted mid-stage — resume and finish that one
   (re-read its steps, inspect the working tree to see what's already done) before
   starting anything new. Otherwise, take the first **PENDING** stage.
2. **Read the stage file** -- follow the link in the tracker to the stage's .md file.
3. **Read resources** -- if the stage references shared resources, find them in `resources/`.
4. **Resolve ambiguity yourself** -- there is no user to ask during an autonomous run.
   Pick the most reasonable interpretation that fits the codebase and existing
   conventions, record it under **Decisions**, and proceed. Only defer to the Backlog
   if the ambiguity genuinely blocks any sensible implementation.
5. **Implement** -- execute the steps described in the stage.
6. **Validate** -- run the verification checks and the test suite. **If anything fails,
   do not stop — triage it via the self-healing loop below.**
7. **Update this index** -- mark the stage DONE in the progress tracker, add brief notes
   about what was done and any deviations. Log every problem you hit in **Fixed Issues**
   (if resolved) or **Backlog** (if deferred). Never silently drop a problem.
8. **Commit** -- create an atomic commit with the message specified in the stage.
   Include all changed files (code, config, docs, and this plan's index.md).

Repeat until every stage is DONE or terminally deferred. After the last stage, **sweep
the Backlog**: attempt any items that are now resolvable, and leave the rest for a
follow-up run.

### Mandatory stop

These override the autonomy rule. They exist because the actions are costly and outward-facing.

- **Stage 6 requires explicit human authorisation before `sbatch`.** Submitting spends roughly
  26 GPU-hours of the `plgcountercontex-gpu-gh200` grant. Run every preflight check, print the
  exact commands and the `sbatch --test-only` output, then stop and wait. Never submit
  unattended.

### Self-healing loop (handling problems)

When a step fails — failing test, build/lint/type error, a bug in the new code, an
unexpected runtime error:

1. **Triage** the problem as *light* or *heavy*.
   - **Light** -- self-contained and fixable in a focused effort: a failing unit test,
     a lint/type error, a missing import, a small logic bug in code you just wrote.
   - **Heavy** -- needs an architectural decision, spans many files, depends on an
     external blocker, contradicts the plan's assumptions, or has already survived a
     fix attempt.
2. **Light → delegate the fix to a subagent.** Spawn a focused subagent (Agent/Task
   tool) with: the failing command and its full output, the relevant file paths, the
   stage goal, and a crisp deliverable (e.g. "make `<test>` pass without weakening
   assertions"). Delegating keeps the main execution context clean. Re-run verification
   when it returns. Cap at **2 attempts per issue** — if still failing, treat it as heavy.
3. **Heavy → defer to the Backlog.** Add a self-contained entry (see the Backlog table).
   Do **not** keep grinding and do **not** halt the plan.
4. **Decide the stage's disposition:**
   - If the stage's core goal is met without the deferred item → mark **DONE**, note the
     backlog reference, and continue.
   - If the deferred item is essential to this stage → mark **BLOCKED**, note the backlog
     reference, and continue to the next *independent* stage. Only stop the run when every
     remaining stage depends on blocked work.
5. **Record** the outcome: resolved problems → **Fixed Issues**; deferred problems → **Backlog**.

### Guardrails

- Keep every commit in a working, buildable state.
- **Never weaken, skip, or delete a test to make it pass.** If a test is genuinely wrong,
  fix it correctly and note it in Fixed Issues.
- Never use `git commit --no-verify`.
- Don't expand a stage's scope to chase a heavy problem — that's what the Backlog is for.
- **Never read, print, echo, or transfer `.env`.** It contains `PLG_PASSWORD`. Cluster access
  is key-based only. Read non-secret selection variables with a targeted grep that excludes
  the password key. Every `rsync` to the cluster must carry `--exclude '.env'`.
- **Never run `uv sync` against the Helios environment.** See Context.
- **Never run two configs as concurrent Slurm jobs.** `experiments/_common.py::append_table`
  does an unlocked read-modify-write on a shared CSV. Chain with `--dependency=afterany`.
- **Never write submission metadata inside the cluster Git checkout.** Store run-2 job IDs at
  `$HEAVY/run2/run2.jobids` and copy them into this index; an untracked file under `slurm/`
  fails the dependent jobs' source-clean preflight.
- Preserve classifier `validity` as the target-class rate. Preserve `no_cf_found`, `vacuous`,
  and `frac_degenerate` as separate diagnostics; do not fold them into validity. Preserve
  `cf_faith_*_hard_valid` as a joint rate rather than silently redefining it as conditional.
- Phases 01 and 02 write into the repo's tracked `results/` regardless of `--out-dir`
  (`experiments/_common.py:64`, `RESULTS_DIR` is not env-overridable). Expect local
  experiments to dirty tracked files; restore with `git checkout` or work on a copy.

---

## Fixed Issues

Problems encountered during execution and resolved (in place or via a fix subagent).
Leave empty until execution surfaces something.

| # | Stage | Symptom | Root Cause | Resolution | Fixed By |
|---|-------|---------|-----------|------------|----------|

---

## Backlog (Deferred Issues)

Problems deferred for later — too heavy to fix inline without derailing the plan.
Each entry must be **self-contained enough for a future run to pick it up cold**:
state the symptom, where it came from, and a concrete lead for resolving it.

| # | Title | Origin Stage | Severity | Why Deferred | Suggested Next Step | Status |
|---|-------|--------------|----------|--------------|---------------------|--------|
| 1 | Multi-seed campaign (D2) | planning | **high** | Explicit scope decision 2026-08-21: run 2 repeats the single-seed shape for cost. Validity is bimodal (median seed-SD 0.110, p90 0.520; 4 archived cells straddle 0.0/1.0), so run 2's table still carries no error bars. | Successor plan: 10 seeds x `full`+`full_nl` at `n_cf=50`, ~120 GPU-h, via `08_aggregate_and_report.py seeds` (hierarchical bootstrap). Seed variance beats within-seed SE 2-5x, so trade `n_cf` for seeds. Report CARLA/PearlCARLA validity as a Bernoulli mixture (k of 10 seeds), not mean +/- CI. | OPEN |
| 2 | Axis A is tautological (D1) | planning | **high** | Out of the agreed scope. SHD/LagAcc compare the graph to itself; the files say so in their own `note` field. The real corruption + `dynotears` sweep exists (`results/tables/table_graph_quality_full_nl_vs_smoke_spring.csv`, 3 seeds) but never ran on Helios. | Either run `08_aggregate_and_report.py graph_quality` in a future job, or strike Axis A from the paper's claims. Until then, do not present SHD=0 / LagAcc=1.000 as a result. | OPEN |
| 3 | `cf_faith_*` cannot rank (D3) | planning | med | Requires redesigning the metric, not fixing a bug. `rollout_hard` equals 1.0 iff the method is CARLA or OracleCF-Rollout, on all 5 datasets; the soft variant ties the oracle to 4.6e-07. | Present it as a semantics-admission gate, never as a ranking column. A genuinely graded faithfulness metric is a research task. | OPEN |
| 4 | PNS coverage and dead oracle contrast (D6) | planning | med | Out of scope. No `pns.json` for `smoke_nl` or `smoke_spring` (57.9% block coverage). `B_model_oracle == C_world_oracle` in 19/20 cells, so `pns_PS_delta_outcome` is 0.0 in 19/20. `PN_world` is null everywhere. | Run phase 07 `pns` for the smoke tier; separately investigate why the model-vs-world channel is empty — likely the two oracles are constructed identically. | OPEN |
| 5 | Axis B is uncontrolled at one seed (D7) | planning | med | Inherited from the single-seed decision. `causaltemp_xai/eval.py:326-331` states multi-seed CIs are required to control Shift-VR regeneration noise. The `MIN_VALIDITY_BASE_FOR_RATIO=0.3` suppression is also correlated with the metric under study — only methods that already win on validity get a robustness score. | Resolve together with Backlog #1. Until then, do not rank on Shift-VR; consider `--skip-aux` to halve phase 03 cost if Axis B is not being claimed. | OPEN |
| 6 | `scm_noise_plausibility` has no ceiling (D9) | planning | low | Small but needs a design call. `axis_c.py:253-280` documents 1.0 as the Pearl oracle's value; measured 0.87-0.98, and PearlCARLA (0.9779) and CftsCels (0.9777) beat the oracle (0.9749) on `full`. Likely benign finite-sample bias in `mean|eps|`. | Add a `Factual` pseudo-method row so the metric's practical ceiling is measured, not assumed. ~20 LoC in phase 05. | OPEN |
| 7 | `table_axis_c_cf_faith.csv` duplicates (D11) | planning | low | Cosmetic; run-1 report section 8.8 already flags it. 54 rows, 45 unique keys; the whole `full` block appears twice (n=3 profiling and n=100 production) with divergent values. | Truncate the table before run 2, or key `append_table` on `(benchmark, classifier, method, n)` so profiling rows cannot shadow production rows. | OPEN |
| 8 | Config coverage regression (D13) | planning | med | Out of scope. 8 of 13 registered configs never ran on Helios: `real_basicmotions`, `real_epilepsy` (the only real-data evidence), `full_interior_label`, `full_interior_label_late` (the RISK-19 ablation `config.py:20-31` says exists to make H8 falsifiable), `smoke_gaussian`, `smoke_nonmonotonic`, `smoke_regime`, `smoke_regime_hmm`, `full_sparse`. Method `CausalFeasibility` and `CftsCounts` also absent. | Decide which are load-bearing for the paper and schedule a third run. The interior-label ablation shows the most extreme bimodality, so it needs multi-seed too. | OPEN |
| 9 | The task is trivial by construction (D16) | planning | med | Design-level, not a bug. `terminal_threshold` labels on a median split of `x[T-1,0]`, and the LSTM head reads `hn[-1]` (`lstm.py:107-113`) — the classifier reads essentially the scalar the label is defined on. Test accuracy 0.94-0.9985, `full_nl` val = 1.0000. `validity` reduces to "did you move `x[T-1,0]` across the median". | Caps what any validity-based ranking can mean. Either say so plainly in the paper, or add an interior-label config to the headline set (see Backlog #8). | OPEN |
| 10 | CPU/GPU ping-pong in the recourse loop | planning | low | Performance only, no correctness impact. `delta` (`carla.py:277`, `:500`), `mask`, `x_t` and `eps` are built on CPU with no `device=`; `torch_logits` then does `x.to(self.device)` every one of 300-500 steps. Probable contributor to phase-03 runtime. | Fold into stage 1 only if it is a trivial device-placement change; otherwise leave for a performance follow-up. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`. When an item is resolved, flip its
status and summarize the fix in **Fixed Issues**. Heavy items may warrant their own
follow-up plan — link it here.

---

## Decisions

**2026-08-21 — Reduced scope with CARLA fix restored.** The user removed the errata/repository-
hygiene and reachability stages, then restored the CARLA/PearlCARLA implementation stage. The
plan fixes recourse plus D4/D5/D8, tests those contracts, hardens the harness, and reruns. It uses
the recorded regenerated `smoke_nl` reachability evidence instead of a separate full-scale gate.

**2026-08-21 — Rerun shape: repeat the single-seed run-1 shape.** `full` + `full_nl`,
`n_cf=100`, one seed, ~26 GPU-hours. Rationale: cheapest end-to-end validation of the recourse
and reporting fixes. Explicitly accepted cost: run 2 inherits D2 and produces no error bars.
Rejected: 3 seeds (~40 GPU-h) and 10 seeds (~120 GPU-h) — both deferred to Backlog #1 as the
successor plan, to run once the fix itself is known good.

**2026-08-21 — Validity remains classifier-only.** Keep `validity` as the classifier's
target-class rate. Do not add a composite recourse-validity metric. `no_cf_found`, vacuity, and
degeneracy remain separate diagnostics.

**2026-08-21 — Preserve joint faithfulness-validity; add conditional faithfulness.** The existing
`cf_faith_*_hard_valid` keys are the tested joint rate `P(hard-faithful AND valid)` and remain
unchanged. New `cf_faith_*_hard_given_valid` keys report `P(hard-faithful | valid)` and are NaN
only when the valid set is empty. Table provenance lists every suppressed conditional cell.

**2026-08-21 — Do-complexity uses an explicit Pearl-scorable denominator.** Emit the all-instance
mean, the mean conditional on a non-empty Pearl intervention schedule, and `n_do_scorable`.
`frac_vacuous` remains a separate noiseless-semantic diagnostic; it is not the complement of the
Pearl-scorable fraction, as `full_nl/CARLA` demonstrates.

**2026-08-21 — Run-2 publication scope and provenance.** The publication table contains exactly
`full` and `full_nl`; smoke runs are validation diagnostics and are never mixed into the table by
auto-discovery. `git_dirty` means source/code dirt under a pathspec that excludes generated or
deliberately omitted result paths; unfiltered state is preserved separately as
`git_worktree_dirty`. Helios runs two config jobs plus an explicit reports job, accumulates phase
failures into a final non-zero exit, and stores job IDs outside the Git checkout.
