# Plan: Recourse-layer fix and second Helios run

**Date**: 2026-08-21
**Branch**: `run-full-experiments`
**Predecessors**: `docs/full_run_2026-08-18.md` (run 1 report), `docs/general_plan.md`
**Goal**: Fix the shared delta-collapse bug in `CARLARecourse`/`PearlCARLARecourse`, make the
shipped metric tables self-consistent, and re-run `full` + `full_nl` on Helios GH200 to
produce a (dataset, method, metric) table whose every cell can be defended.

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

Four phases. Nothing touches the cluster until the fix is proven locally.

**Phase A — make the record honest (stage 1).** Publish an errata on the run-1 report before
any code changes land, so no point in git history carries unretracted claims. Resolve the
`results/` vs `results_helios/` split, close the `.npy` gitignore gap, retire the misleading
sbatch, and commit the untracked scripts that run 1 actually used.

**Phase B — prove the fix is possible, then make it (stages 2-5).** Stage 2 is a hard GO/NO-GO
gate: measure whether the decision boundary is reachable at `full_nl` scale (75 contraction
steps) as it demonstrably is at `smoke_nl` scale (22 steps). If it is not, no `lam_prox` fix
helps and the benchmark config itself needs redesign — the plan branches rather than burning
GPU-hours on an unreachable target. Stages 3-5 then fix the recourse layer, lock the fix in
with regression tests and CI coverage, and repair the metric-reporting defects.

**Phase C — make the run reproducible (stages 6-7).** Turn the prose recipe into committed,
`--test-only`-validated sbatch files carrying the threading fix and preflight gates. Then run
the entire smoke tier locally end to end as the final gate before spending grant hours.

**Phase D — run and report (stages 8-9).** Submit the chained Helios jobs, collect, rebuild the
table, verify it against an explicit defect checklist, and write the superseding report.

**Rerun shape (decided 2026-08-21):** repeat the run-1 shape — `full` + `full_nl` at
`n_cf=100`, single seed, ~26 GPU-hours. This confirms the fix end to end at full scale for
minimum cost. It does **not** address D2: the resulting table still has no error bars and no
defence against the bimodality. That is an accepted, documented limitation of run 2, and the
multi-seed campaign is recorded in the Backlog as the natural successor plan.

---

## Success Criteria

| Metric | Baseline (run 1) | Target (run 2) | Rationale |
|---|---|---|---|
| CARLA `delta` at `t0`, `full_nl` | 6.4e-08 | > `INTERVENTION_TOL` (1e-3) on >= 90% of instances | Below tolerance the CF is not an intervention at all |
| PearlCARLA `frac_degenerate`, `full_nl` | 1.00 | <= 0.10 | Degenerate CFs are unscorable; the method contributes nothing |
| CARLA raw `validity`, `full_nl` | 0.000 | > 0.30 | Must clear `MIN_VALIDITY_BASE_FOR_RATIO` so the existing Axis B calculation is computable |
| CARLA `recourse_validity`, `full_nl` | 0.000 | > 0.30 | Counts a classifier flip only when a real, non-vacuous intervention was found |
| CARLA `frac_vacuous`, `full_nl` | 1.00 | <= 0.20 | A vacuous CF is mechanism continuation, not recourse |
| CARLA at `delta=0`, `full` | raw validity flips 100/100 | `validity` may remain 1.0, but `recourse_validity == 0`, `no_cf_found == 1`, and the row earns no recourse credit | Preserve classifier validity while rejecting noise-deletion as recourse |
| `do_complexity` naming collision | all-instance and Pearl-scorable values ship under near-identical names | `mean_all` + `mean_pearl_scorable`, with `n_do_scorable` and no claim tying them to `frac_vacuous` | D4 |
| Conditional faithfulness on empty valid set | no separately named conditional metric | `*_hard_given_valid` is NaN and suppressed; existing `*_hard_valid` joint metric remains 0.0 | D5 |
| Oracle metric coverage | 140/350 = 40% | raw `validity`, `recourse_validity`, both explicit do-complexity means and denominator diagnostics present for both oracles; `OracleCF-Pearl` Pearl-scorable mean == 1.0 | D8 |
| Nonlinear path in CI | 0 tests | >= 1 regression test failing on a delta-collapse, running in CI | Prevents silent recurrence |
| Run-1 report | claims 7.2 / 7.4 unretracted | dated errata block at the top | Honest git history |
| Reproducibility | recipe is prose only | committed sbatch, `sbatch --test-only` clean | Run 3 must be one command |

---

## Files That May Be Changed

### Recourse layer (stages 3-4)
- `causaltemp_xai/methods/counterfactual/carla.py` -- proximity-penalty scaling at `:288` and
  `:511`; candidate tie-break at `:302` and `:525`; sub-tolerance no-op guard after both loops;
  constructor defaults at `:181-184` and `:394-397`.
- `causaltemp_xai/config.py` -- optional per-config recourse hyperparameters.
- `experiments/03_run_cf_methods.py` -- `build_methods()` at `:86-109`, the sole call site that
  overrides method hyperparameters; persist `no_cf_found_<Method>.npy` sidecars.
- `experiments/04_evaluate_axes.py`, `experiments/_common.py` -- load no-CF sidecars, emit
  `recourse_validity`, preserve the joint faithfulness-validity metrics, add the separately
  named conditional metrics, and define code-clean provenance.
- `tests/test_methods.py` -- add a nonlinear (`MLPMechanism`) fixture and regression tests.
- `.github/workflows/ci.yml` -- add `smoke_nl` to the pipeline smoke harness at `:70-78`.

### Metric and reporting layer (stage 5)
- `causaltemp_xai/eval.py` -- `evaluate_method()` conditional-metric NaN handling (D5),
  do-complexity dilution (D4).
- `experiments/05_run_oracle_control.py` -- emit the missing oracle metrics (D8).
- `experiments/make_final_table.py` -- preserve joint metrics, add conditional-metric
  suppression provenance, distinguish the two Pearl-semantic do-complexity denominators, and
  require explicit publication configs.
- `tests/test_eval.py`, `tests/test_do_complexity.py` -- cover the new behaviour.
- `tests/test_metric_adversarial.py` -- preserve the joint metric contract and cover the new
  conditional metric and recourse-validity semantics.

### Harness (stages 1, 6)
- `slurm/helios_full_run.sbatch` (new), `slurm/helios_smoke.sbatch`,
  `slurm/helios_reports.sbatch` (new), `slurm/helios_setup_env.sbatch`,
  `slurm/helios_import_check.sbatch` -- commit + threading fix and accumulated failure status.
- `slurm/full_pipeline_rerun.sbatch` -- retire.
- `.gitignore` -- `.npy` coverage for any results tree.

### Documentation (stages 1, 9)
- `docs/full_run_2026-08-18.md` -- errata block only; body left as the historical record.
- `docs/full_run_2026-08-2X.md` (new) -- the superseding report.
- `docs/risk_register.md` -- close RISK-17.

### Deleted
- `experiments/results.json`, `experiments/per_instance.csv`, `experiments/results_archive/` --
  pre-retune artifacts carrying the superseded `gain=0.8` nonlinear hyperparameters.

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Errata and repo hygiene](stages/01-errata-and-hygiene.md) | PENDING | | |
| 2 | [Reachability gate (GO/NO-GO)](stages/02-reachability-gate.md) | PENDING | | |
| 3 | [Fix the recourse layer](stages/03-fix-recourse-layer.md) | PENDING | | |
| 4 | [Regression tests and CI coverage](stages/04-regression-tests-ci.md) | PENDING | | |
| 5 | [Metric reporting fixes](stages/05-metric-reporting-fixes.md) | PENDING | | |
| 6 | [Helios job scripts](stages/06-helios-job-scripts.md) | PENDING | | |
| 7 | [Local full validation](stages/07-local-full-validation.md) | PENDING | | |
| 8 | [Cluster preflight and submit](stages/08-cluster-submit.md) | PENDING | **HUMAN GATE** | |
| 9 | [Collect, table, document](stages/09-collect-table-document.md) | PENDING | | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

---

## Execution Protocol

This plan is built for **autonomous, unattended execution**, with two explicit exceptions
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

### Two mandatory stops

These override the autonomy rule. They exist because the actions are costly and outward-facing.

- **Stage 2 is a decision gate.** If the reachability probe returns NO-GO, do not proceed to
  stage 3. Record the measurement, mark stages 3-9 BLOCKED, write the finding into the
  Backlog, and stop. A NO-GO means the `full_nl` config is unreachable by any proximity-
  penalised optimiser and needs redesign — a decision no autonomous run should make alone.
- **Stage 8 requires explicit human authorisation before `sbatch`.** Submitting spends roughly
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
- Preserve raw classifier `validity`. Use `recourse_validity` for claims about successful
  recourse, and preserve `cf_faith_*_hard_valid` as a joint rate rather than silently
  redefining it as conditional.
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
| 10 | CPU/GPU ping-pong in the recourse loop | planning | low | Performance only, no correctness impact. `delta` (`carla.py:277`, `:500`), `mask`, `x_t` and `eps` are built on CPU with no `device=`; `torch_logits` then does `x.to(self.device)` (`lstm.py:425`) every one of 300-500 steps. Probable contributor to the phase-03 runtime. | Construct the whole rollout on `model.device`. ~6 LoC. Fold into stage 3 only if it costs nothing; otherwise a standalone perf change. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`. When an item is resolved, flip its
status and summarize the fix in **Fixed Issues**. Heavy items may warrant their own
follow-up plan — link it here.

---

## Decisions

**2026-08-21 — Scope: recourse layer plus reporting fixes.** Fix `carla.py` *and* the D4/D5/D8
metric-reporting defects, rather than the recourse layer alone. Rationale: a table whose
`do_complexity` column silently means two different things, and whose `*_hard_valid` column
cannot distinguish "faithless" from "never valid", is not defensible even if every method
produces a real counterfactual. Rejected: recourse-only (leaves the tables misleading); and
recourse + reporting + restored coverage (D1 graph-quality, D6 PNS, D13 configs) — deferred to
Backlog because it multiplies cluster cost before the primary fix is proven.

**2026-08-21 — Rerun shape: repeat the single-seed run-1 shape.** `full` + `full_nl`,
`n_cf=100`, one seed, ~26 GPU-hours. Rationale: cheapest end-to-end confirmation that the fix
works at full scale. Explicitly accepted cost: run 2 inherits D2 and produces no error bars.
Rejected: 3 seeds (~40 GPU-h) and 10 seeds (~120 GPU-h) — both deferred to Backlog #1 as the
successor plan, to run once the fix itself is known good.

**2026-08-21 — Run-1 report: errata now, supersede later.** Add a dated errata block to
`docs/full_run_2026-08-18.md` in stage 1, before any code changes, then write a fresh report in
stage 9 and link the two. Rationale: the doc is already committed and pushed to
`fork/run-full-experiments`; leaving sections 7.2 and 7.4 unretracted through the whole fix
cycle would put unflagged false claims in the repo's history. The body stays intact as the
historical record of run 1.

**2026-08-21 — Stage 2 exists as a hard gate.** The reachability sweep that proved the boundary
is crossable was run only at `smoke_nl` scale (22 contraction steps). `full_nl` has 75. Because
the mechanism contracts per step, the delta required at full scale may exceed any sane
proximity budget, in which case rescaling `lam_prox` cannot help and the config needs redesign.
Measuring this costs minutes; discovering it after a 26 GPU-hour run costs a day.

**2026-08-21 — Reachability is a joint-delta property.** A coordinate sweep remains as a
reproducibility check, but it can never justify NO-GO by itself because both recourse methods
optimise a joint `k`-dimensional delta. Stage 2 now requires an unpenalised joint search with
multiple starts and persists the probe and output. NO-GO requires both probe families to fail.

**2026-08-21 — Raw validity and recourse validity are separate.** Keep `validity` as the
classifier's raw target-class rate. Add `recourse_validity`, which requires a classifier flip,
an algorithm-level `no_cf_found=false`, and a non-vacuous intervention. A zero-delta noiseless
rollout can remain classifier-valid because deleting noise changes the trajectory, but it earns
zero recourse credit. CARLA/PearlCARLA persist their no-CF status in per-method sidecars so this
decision survives Phase-03/04 serialization.

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
