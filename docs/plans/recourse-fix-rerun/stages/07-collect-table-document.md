# Stage 7: Collect, table, document

**Goal**: Pull run 2, rebuild the table, verify it against the defect checklist, and document
the corrected recourse and metric/reporting outputs.
**Dependencies**: Stage 6; all jobs finished.

---

## Steps

1. **Confirm the jobs actually succeeded.** Successful submission is not successful execution.
   For every job in the chain — not only the last one:
   ```
   sacct -j <id> --format=JobID,JobName,Partition,Account,State,Elapsed,AllocTRES,ReqMem,ExitCode
   ```
   Require `COMPLETED` and `ExitCode 0:0`, and read the `.err` files for hidden failures. A job
   that exits `0:0` with missing output is incomplete, not successful.

2. **Pull the results** to `results/` on the laptop, excluding nothing that the table needs.
   Confirm `summary.json`, `pns.json` and `shift_vr.json` exist and are non-empty for both
   configs, and that their provenance shows the run-2 commit with source-scoped
   `git_dirty: false`. `git_worktree_dirty` may be true because generated results are tracked or
   deliberately omitted; report both fields and never substitute the latter for code dirt.

3. **Rebuild the run-2 publication table.** Run exactly:
   ```bash
   .venv/bin/python experiments/make_final_table.py --configs full full_nl
   ```
   Confirm the provenance JSON names exactly those two configs and the run-2 commit. Never use
   unfiltered auto-discovery here: mixing retained smoke artifacts into the run-2 table violates
   the publication-scope check. Stage-5 smoke tables remain separate validation diagnostics.

4. **Run the defect checklist** in `resources/checklist.md` against the new full table. This is
   the gate that decides whether run 2 is publishable. Every item passes or is waived in writing.

5. **Verify each success criterion** from the index against the `full_nl` numbers specifically.
   Keep `validity` classifier-only and inspect `frac_no_cf_found`, `frac_vacuous`, and
   `frac_degenerate` as separate diagnostics of whether the CARLA fix produced real actions.

6. **Diff run 2 against run 1** cell by cell, and attribute every material change to the metric,
   oracle, table, or harness changes in this plan. Pay attention to the **linear `full` tier**:
   CARLA `validity` may remain high because it reports classifier outcome only. Report
   `frac_no_cf_found` and `frac_vacuous` beside it without redefining validity.

7. **Write the superseding report**, `docs/full_run_<date>.md`. Follow the structure of
   `docs/full_run_2026-08-18.md` (provenance and caveats, compute, commands, datasets, method
   hyperparameters, results tables, findings, known limitations, next steps, artifacts) and
   its ASD-STE100 Simplified Technical English style. It must state plainly:
   - What the delta-collapse bug invalidated in run 1 and how stage 1 fixed it.
   - That run 2 is **single-seed** and therefore carries no error bars, and that validity is
     known to be bimodal across seeds (median seed-SD 0.110, p90 0.520) — so no ranking in this
     table should be read as established. Point at Backlog #1.
   - That Axis A remains tautological (Backlog #2) and must not be presented as structure
     recovery.
   - That `cf_faith_*` is a semantics-admission gate, not a quality ranking (Backlog #3).
   - That Axis B is uncontrolled at one seed by the code's own stated requirement (Backlog #5).

8. **Update the memory notes** so a future session starts from the corrected recourse and metric
   semantics and the fact that run 2 is single-seed by decision rather than by oversight.

---

## Verification

- [ ] Every job in the chain shows `COMPLETED` / `0:0`, and the `.err` files are clean.
- [ ] `results/full/lstm/summary.json` and `results/full_nl/lstm/summary.json` exist, are
      non-empty, and their provenance shows the run-2 commit with source-scoped
      `git_dirty: false`; `git_worktree_dirty` is also recorded.
- [ ] `results/tables/final_table_long.csv` covers exactly `full` and `full_nl`, with all seven
      methods plus both oracles; it contains no smoke or pre-run rows.
- [ ] Every item in `resources/checklist.md` passes or is waived with a written reason.
- [ ] Every success criterion in the index is met, or its miss is documented as a finding.
- [ ] The run-1 vs run-2 diff is complete and every material change is attributed.
- [ ] The new report exists and distinguishes classifier validity from no-CF and vacuity diagnostics.
- [ ] The Backlog in the index is swept: any item now resolvable is attempted, the rest carried.

---

## Commit

`[docs] record the run-2 Helios benchmark on the fixed recourse layer`
