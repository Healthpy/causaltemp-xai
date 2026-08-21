# Stage 3: Regression tests

**Goal**: Pin the CARLA delta fix and corrected D4, D5, and D8 contracts.
**Dependencies**: Stages 1-2.

---

## Steps

1. **Pin the CARLA/PearlCARLA fix on nonlinear data.**
   - File: `tests/test_methods.py`.
   - Build a small current-hyperparameter `NlinearSCMT` fixture.
   - Assert both methods produce an intervention-row delta above `INTERVENTION_TOL` on at least
     90% of cases and do not silently prefer the smallest failed candidate.
   - Assert `generate_batch_with_status()` emits a Boolean status vector and marks a forced
     zero-delta failure without changing the ndarray-only legacy APIs.
   - Assert Phase 03 writes matching sidecars and Phase 04 rejects a missing, wrong-dtype, or
     wrong-length CARLA/PearlCARLA sidecar while leaving classifier validity unchanged.

2. **Pin the two do-complexity denominators.**
   - Files: `tests/test_do_complexity.py`, `tests/test_metric_adversarial.py`.
   - Assert that `do_complexity_mean_all` includes every instance and
     `do_complexity_mean_pearl_scorable` includes only non-empty Pearl schedules.
   - Assert `mean_all == mean_pearl_scorable * n_do_scorable / n` when scorable and NaN when
     `n_do_scorable == 0`.
   - Include the `full_nl/CARLA` counterexample showing that noiseless `frac_vacuous` is not the
     complement of the Pearl-scorable fraction.

3. **Preserve joint faithfulness-validity and test the new conditional metric.**
   - Files: `tests/test_eval.py`, `tests/test_metric_adversarial.py`.
   - Keep `cf_faith_*_hard_valid = mean(hard_i * valid_i)` unchanged.
   - Cover zero, partial, and full valid sets for `cf_faith_*_hard_given_valid`; only the empty
     valid set produces NaN.

4. **Test oracle coverage.**
   - Exercise `experiments/05_run_oracle_control.py` with a classifier checkpoint.
   - Assert both oracle rows contain classifier validity, both do-complexity means, and
     denominator diagnostics.
   - Assert `OracleCF-Pearl.do_complexity_mean_pearl_scorable == 1.0`.

5. **Test publication filtering and provenance.**
   - Assert `make_final_table.py --configs full full_nl` excludes retained smoke directories.
   - Assert conditional NaN suppressions are enumerated in `final_table_provenance.json`.
   - Assert result-tree changes leave source-scoped `git_dirty=false` while a source edit makes
     it true; preserve unfiltered state in `git_worktree_dirty`.

---

## Verification

- [ ] `uv run pytest tests/test_methods.py tests/test_eval.py tests/test_do_complexity.py tests/test_metric_adversarial.py -q` passes.
- [ ] The table-builder and provenance tests pass with retained smoke fixtures present.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `git diff --check` passes.

---

## Commit

`[test] pin the CARLA fix, metric semantics, and publication filtering`
