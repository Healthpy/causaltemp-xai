# Defect checklist

Run this against any newly generated `final_table_long.csv` before calling a run usable. Every
item passes, or is waived in writing in the index.

## Primary — the delta collapse

- [ ] **C1** NoiselessSCMRecourse's intervention-row delta exceeds `INTERVENTION_TOL` on at least 90% of
      instances for every config.
- [ ] **C2** No NoiselessSCMRecourse row has `frac_vacuous > 0.20`.
- [ ] **C3** No PearlSCMRecourse row has `frac_degenerate > 0.10`.
- [ ] **C4** `n_cf_faith_scorable > 0` for NoiselessSCMRecourse and PearlSCMRecourse on every config.
- [ ] **C5** A forced zero-delta counterfactual may retain classifier `validity`, while
      `no_cf_found=1` and `vacuous=1` are reported separately. No composite validity metric is
      introduced.

## Metric self-consistency

- [ ] **C6** `do_complexity_mean_all`, `do_complexity_mean_pearl_scorable`, and
      `n_do_scorable` are present, and when `n_do_scorable > 0`,
      `mean_all == mean_pearl_scorable * n_do_scorable / n`. When it is zero, the scorable mean
      is NaN. `frac_vacuous` is separate and is never used as this denominator.
- [ ] **C7** `*_hard_valid` remains the joint rate and reads `0.0` where `validity == 0`.
      The separately named `*_hard_given_valid` is NaN, suppressed from the table, and listed
      in suppression provenance when the valid set is empty.
- [ ] **C8** No inf or out-of-range value: no fraction outside `[0, 1]`, no negative distance or
      count, and no `do_complexity_mean > T`. NaN appears only where the documented denominator
      is empty and is recorded in suppression provenance.
- [ ] **C9** `n_cf_faith_scorable == n * (1 - frac_degenerate)` exactly in every cell.
- [ ] **C10** `validity` remains solely classifier-derived; `no_cf_found`, `vacuous`, and
      `frac_degenerate` remain separate diagnostics.

## Oracles as controls

- [ ] **C11** Both oracles carry classifier `validity`, both explicit do-complexity means, and
      their denominator diagnostics.
- [ ] **C12** `OracleCF-Pearl` has `do_complexity_mean_pearl_scorable == 1.0`. It performs exactly
      one `do()` by construction (`experiments/_common.py:613-632`).
- [ ] **C13** `OracleCF-Pearl` ranks first on `cf_faith_pearl_soft`, and `OracleCF-Rollout` first
      on `cf_faith_rollout_soft`, on every config.
- [ ] **C14** The report states that oracles are not expected to lead on `proximity_l1`; they use
      a fixed `ORACLE_SHIFT = 1.5` and do not optimise proximity.

## Coverage

- [ ] **C15** Every intended (dataset, method) pair is present. Account for every gap as a
      method skip or a documented downstream suppression.
- [ ] **C16** `shift_vr` suppressions are enumerated with `validity_base`, and the report states
      that `MIN_VALIDITY_BASE_FOR_RATIO = 0.3` correlates suppression with the quantity studied.
- [ ] **C17** `results/tables/table_axis_c_cf_faith.csv` has no duplicate
      `(benchmark, classifier, method)` keys.
- [ ] **C18** Classifier test accuracy is recorded for every config and clears `target_acc=0.9`.

## Provenance

- [ ] **C19** `final_table_provenance.json` names exactly `full` and `full_nl` plus the expected
      commit, with source-scoped `git_dirty: false`; it records unfiltered
      `git_worktree_dirty` separately.
- [ ] **C20** The publication table contains exactly `full` and `full_nl`, both from run 2. No
      result predates the run; timestamps and provenance match the job windows. Smoke artifacts
      remain diagnostics and are not auto-discovered into this table.

## Claims that must be qualified, not fixed

- [ ] **C21** The run uses one seed and therefore establishes no ranking; validity is known to be
      bimodal across seeds (median seed-SD 0.110, p90 0.520). See Backlog #1.
- [ ] **C22** Axis A compares the graph to itself; SHD = 0 and LagAcc = 1.000 are not structure
      recovery. `cf_faith_*` is a semantics-admission gate, not a quality ranking.
- [ ] **C23** Axis B is uncontrolled at one seed, and `scm_noise_plausibility` has no measured
      factual ceiling. See Backlog #5 and #6.
