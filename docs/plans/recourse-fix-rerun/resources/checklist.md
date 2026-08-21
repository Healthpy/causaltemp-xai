# Defect checklist

Run this against any newly generated `final_table_long.csv` before calling a run usable. Every
item passes, or is waived in writing in the index. Each check names the run-1 value it is
guarding against.

## Primary — the delta collapse

- [ ] **C1** No method has `frac_degenerate > 0.10` on any config.
      *Run 1: PearlCARLA = 1.00 on `full_nl` and `smoke_nl`.*
- [ ] **C2** No method has `frac_vacuous > 0.20` on any config.
      *Run 1: CARLA = 1.00 on `full_nl` and `smoke_nl`; CftsConfeti = 0.85 on `smoke_spring`.*
- [ ] **C3** For CARLA and PearlCARLA, the median `|x_cf - x|` at the derived intervention row
      exceeds `INTERVENTION_TOL = 1e-3` on every config.
      *Run 1: 6.4e-08 to 2.9e-02.*
- [ ] **C4** `n_cf_faith_scorable > 0` for every (config, method).
      *Run 1: 0 for PearlCARLA on `full_nl` and `smoke_nl`.*
- [ ] **C5** A zero-delta counterfactual may retain raw classifier `validity`, but it has
      `no_cf_found=1`, `vacuous=1`, and `recourse_validity=0`; it earns no recourse credit.
      *Run 1: 100/100 raw flips on `full` at `t0=25`, so CARLA's `validity = 1.000` was free.*

## Metric self-consistency

- [ ] **C6** `do_complexity_mean_all`, `do_complexity_mean_pearl_scorable`, and
      `n_do_scorable` are present, and when `n_do_scorable > 0`,
      `mean_all == mean_pearl_scorable * n_do_scorable / n`. When it is zero, the scorable mean
      is NaN. `frac_vacuous` is separate and is never used as this denominator.
      *Run 1: only the diluted value shipped under a name suggesting otherwise; CftsConfeti read
      56.0 on `full` while actually touching all 100 timesteps.*
- [ ] **C7** `*_hard_valid` remains the joint rate and reads `0.0` where `validity == 0`.
      The separately named `*_hard_given_valid` is NaN, suppressed from the table, and listed
      in suppression provenance when the valid set is empty.
      *Run 1: 685/685 rows correctly had joint rate 0.0; the conditional statistic was absent.*
- [ ] **C8** No NaN, inf, or out-of-range value: no fraction outside `[0, 1]`, no negative
      distance or count, no `do_complexity_mean > T`.
      *Run 1: clean. Keep it that way.*
- [ ] **C9** `n_cf_faith_scorable == n * (1 - frac_degenerate)` exactly, in every cell.
- [ ] **C10** No vacuous or `no_cf_found` counterfactual receives `recourse_validity` credit;
      raw classifier `validity` remains reported separately.

## Oracles as controls

- [ ] **C11** Both oracles carry raw `validity`, `recourse_validity`, both explicit
      do-complexity means, and their denominator diagnostics.
      *Run 1: coverage 140/350 = 40%; both absent.*
- [ ] **C12** `OracleCF-Pearl` has `do_complexity_mean_pearl_scorable == 1.0`. It performs exactly one
      `do()` by construction (`experiments/_common.py:613-632`), so any other value is a bug in
      the oracle or in the metric.
- [ ] **C13** `OracleCF-Pearl` ranks first on `cf_faith_pearl_soft`, and `OracleCF-Rollout` first
      on `cf_faith_rollout_soft`, on every config.
      *Run 1: held, except `OracleCF-Rollout` ranked 2/9 on `smoke` at n=5.*
- [ ] **C14** Note explicitly that oracles are **not** expected to lead on `proximity_l1` — they
      apply a fixed `ORACLE_SHIFT = 1.5` and do not optimise proximity. Run 1's `OracleCF-Pearl`
      ranked 7/9 there, which is correct, not a defect. Do not describe the oracles as upper
      bounds without this qualification.

## Coverage

- [ ] **C15** Every intended (dataset, method) pair is present. Account for every gap: is the
      method skipped, or is the metric suppressed downstream of a degeneracy gate?
- [ ] **C16** `shift_vr` suppressions are enumerated with their `validity_base`, and it is stated
      that the `MIN_VALIDITY_BASE_FOR_RATIO = 0.3` gate (`causaltemp_xai/eval.py:259`) correlates
      the suppression with the quantity under study.
      *Run 1: 8 of 35 cells null.*
- [ ] **C17** `results/tables/table_axis_c_cf_faith.csv` has no duplicate
      `(benchmark, classifier, method)` keys.
      *Run 1: 54 rows, 45 unique keys — the whole `full` block appeared twice, at n=3 and n=100,
      with divergent values.*
- [ ] **C18** Classifier test accuracy is recorded for every config and clears `target_acc=0.9`.
      *Run 1: 0.940-0.9985. Note that ~1.0 is structural, not a sign of quality — the label is a
      median threshold on `x[T-1,0]` and the LSTM head reads `hn[-1]`.*

## Provenance

- [ ] **C19** `final_table_provenance.json` names exactly `full` and `full_nl` plus the expected
      commit, with source-scoped `git_dirty: false`; it also records unfiltered
      `git_worktree_dirty` separately.
      *Run 1: `git_dirty: true`, explained post hoc by the rsync excluding tracked directories.*
- [ ] **C20** The publication table contains exactly `full` and `full_nl`, both from run 2. No
      result predates the run; check timestamps and provenance against each job's elapsed window.
      Smoke validation artifacts remain diagnostics and are not auto-discovered into this table.

## Claims that must be qualified, not fixed

These are known limitations carried into run 2 by decision. The report must state each one; the
checklist exists so none is forgotten.

- [ ] **C21** Single seed, so no error bars. Validity is bimodal across seeds (median seed-SD
      0.110, p90 0.520; two run-1 point estimates fell outside the archived 3-seed CI). No
      ranking in this table is established. See Backlog #1.
- [ ] **C22** Axis A compares the graph to itself. SHD = 0 and LagAcc = 1.000 are not structure
      recovery. See Backlog #2.
- [ ] **C23** `cf_faith_*` is a semantics-admission gate, not a quality ranking. See Backlog #3.
- [ ] **C24** Axis B is uncontrolled at one seed by `causaltemp_xai/eval.py:326-331`'s own stated
      requirement. See Backlog #5.
- [ ] **C25** `scm_noise_plausibility` has no measured ceiling — no factual control row exists.
      See Backlog #6.
