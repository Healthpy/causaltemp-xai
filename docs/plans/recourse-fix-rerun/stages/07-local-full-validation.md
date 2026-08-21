# Stage 7: Local full validation

**Goal**: Run the entire smoke tier end to end on the fixed code and confirm every success
criterion that does not need full scale — the last gate before spending grant hours.
**Dependencies**: Stages 3, 4, 5, 6.

---

## Steps

1. **Regenerate all three smoke configs from scratch** — `smoke`, `smoke_nl`, `smoke_spring` —
   into a scratch tree. Do not reuse the laptop's checked-in `data/scm_t/`: it is dated
   2026-07-07/08 and carries the pre-retune nonlinear hyperparameters (`gain=0.8`,
   `decay_range=[0.3, 0.8]`, `spectral_cap=0.9`, `init_gain=0.7`), on which the bug does not
   reproduce at all.
   - Regeneration takes about 2 minutes per smoke config and reproduces the Helios classifier
     accuracies exactly (`smoke_nl`: train 0.96 / val 0.98 / test 0.96, matching
     `results/smoke_nl/lstm/train_report.json`). If your accuracies do not match, stop and find
     out why before trusting anything downstream.
   - Expect phases 01 and 02 to write into the tracked `results/` regardless of `--out-dir`
     (`experiments/_common.py:64`). Restore with `git checkout` afterwards, or work on a copy.

2. **Run phases 01 through 05 plus phase 07 `pns`** for each smoke config with the full
   seven-method roster (see `resources/commands.md`), then rebuild the table with
   `experiments/make_final_table.py --configs smoke smoke_nl smoke_spring`. This is a local
   diagnostic table; Stage 9's publication table contains only `full` and `full_nl`.

3. **Check every success criterion from the index that smoke scale can reach.** In particular,
   confirm on `smoke_nl` that CARLA's `frac_vacuous` and PearlCARLA's `frac_degenerate` have
   both dropped from 1.00, that raw `validity` clears `MIN_VALIDITY_BASE_FOR_RATIO = 0.3`
   (`causaltemp_xai/eval.py:259`) so the existing Axis B calculation is computable, and that
   `recourse_validity > 0.30` so those flips are earned by real interventions.

4. **Run the defect checklist** in `resources/checklist.md` against the new smoke table. Every
   item must pass or be explicitly waived with a reason recorded in the index.

5. **Measure the cost delta and re-size the walltime.** Time phase 03 on `smoke_nl` before and
   after the fix. The lambda backoff from stage 3 can multiply optimiser steps on hard
   instances. Extrapolate against the run-1 constant — Phase 03 = `17 + 381.3 * n_cf` seconds
   for `full`, with `full_nl` at 1.3-1.5x — and update the `--time` values in
   `slurm/helios_full_run.sbatch` if the fix has made the run materially slower. A TIMEOUT at
   hour twelve wastes the whole allocation.

6. **Compare against run 1 side by side.** Produce a small diff table: run-1 value vs run-2-local
   value for every `smoke`/`smoke_nl`/`smoke_spring` cell that changed. Every material change
   must trace to a specific fix from stages 3 or 5. An unexplained change is a bug, not a
   result — chase it before proceeding.

---

## Verification

- [ ] All three smoke configs complete phases 01-05 and 07 without error.
- [ ] `smoke_nl`: CARLA `frac_vacuous` <= 0.20, raw `validity` > 0.30, and
      `recourse_validity > 0.30`; PearlCARLA `frac_degenerate` <= 0.10.
- [ ] `smoke`: CARLA still has raw classifier validity; the zero-delta control has
      `no_cf_found=1`, `vacuous=1`, and `recourse_validity=0` even if raw validity is 1.
- [ ] Every item in `resources/checklist.md` passes or is waived with a recorded reason.
- [ ] The run-1 vs run-2 diff table exists and every material change is attributed to a fix.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `slurm/helios_full_run.sbatch` walltimes reflect the measured cost delta.
- [ ] `git status --porcelain` is clean afterwards (the scratch tree is outside the repo, and
      any tracked files phases 01/02 dirtied have been restored).

---

## Commit

`[data] validate the recourse fix across the full smoke tier`
