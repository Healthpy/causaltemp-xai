# Stage 1: Errata and repo hygiene

**Goal**: Publish an errata on the run-1 report and leave the working tree clean enough that
the `git_dirty` provenance flag means something again.
**Dependencies**: None.

---

## Steps

1. **Add a dated errata block to the run-1 report.**
   - File: `docs/full_run_2026-08-18.md`
   - Insert immediately after the title, before section 1. Do **not** edit the body —
     it stays as the historical record of run 1.
   - Content, in ASD-STE100 Simplified Technical English to match the rest of the document:
     state that an audit on 2026-08-21 found the CARLA and PearlCARLA results in sections 7.1,
     7.2 and 7.4 to be artifacts of a single delta-collapse bug; that CARLA's `validity = 1.000`
     on the linear `full` tier is noise-deletion and is reproduced exactly by a zero-delta
     counterfactual (100/100 flips at `t0=25`); that section 7.2's reading of the linear tier as
     a working baseline does not hold; and that section 7.4 is the same bug, not a horizon
     effect. Link to this plan directory and to the successor report once stage 9 names it.

2. **Promote the Helios results to the canonical tree.**
   - Move `results/` (the pre-Helios run at commit `a38b097`) to `results_archive_pre_helios/`
     or delete it if git already holds it — check `git status` first; much of `results/` is
     tracked, so prefer `git mv` for tracked paths over a plain `mv`.
   - Move the contents of `results_helios/` into `results/` so that
     `experiments/make_final_table.py` with its default `--results-dir results` reads run 1's
     numbers rather than the older ones.
   - Rationale: two parallel unmerged trees exist today, and the default code path silently
     reads the stale one. A second run would create a third copy.

3. **Close the `.npy` gitignore gap.**
   - File: `.gitignore`
   - The existing pattern is `results/**/*.npy`, which did not cover `results_helios/`.
     After step 2 the pattern applies again, but add a defensive `results*/**/*.npy` so a
     future parallel tree cannot leak 7.2 MB of counterfactual arrays into a commit.
   - Verify with `git status --porcelain | grep '\.npy'` returning nothing.

4. **Commit the scripts run 1 actually used.**
   - `experiments/make_final_table.py`, `slurm/helios_setup_env.sbatch`,
     `slurm/helios_import_check.sbatch`, `slurm/helios_smoke.sbatch`.
   - These are referenced by `docs/full_run_2026-08-18.md` section 3 but are untracked, so a
     fresh clone cannot regenerate the tables the report describes.

5. **Retire the misleading sbatch.**
   - File: `slurm/full_pipeline_rerun.sbatch`
   - It targets account `tesr123566` on partition `gpu_a100` — a different cluster — and drives
     the pipeline through `uv run`, which on Helios resolves torch 2.12.0 against CUDA 13 and
     risks a silent CPU fallback. Delete it. Its header comments about the CARLA device-placement
     and cuDNN eval-mode fixes are historically useful; move that text into
     `docs/full_run_2026-08-18.md` or a comment in the new Helios script (stage 6) before deleting.

6. **Delete pre-retune artifacts.**
   - `experiments/results.json`, `experiments/per_instance.csv`, `experiments/results_archive/`
   - All sit in the wrong directory (`experiments/` rather than `results/`), are dated 2026-07-08,
     and their embedded config shows `full_nl` with `gain=0.8` — the nonlinear hyperparameters
     replaced on 2026-08-11 because they left the MLP branch carrying only 1.6% of output
     variance. These numbers are provably obsolete.
   - Before deleting `results_archive/`, confirm nothing references it:
     `grep -rn "results_archive" --include='*.py' --include='*.md' .`

7. **Decide on the untracked audit document.**
   - `docs/repository_audit_2026-07-15.md` (297 lines) is distinct content from the tracked
     `docs/critical_review_2026-08-11.md`. Commit it if it should be preserved; delete it
     otherwise. Do not leave it untracked.

---

## Verification

- [ ] `git status --porcelain` is empty except for this plan directory.
- [ ] `git status --porcelain | grep '\.npy'` returns nothing.
- [ ] `python experiments/make_final_table.py --help` runs, and running it with no arguments
      discovers the five run-1 configs (`full`, `full_nl`, `smoke`, `smoke_nl`, `smoke_spring`)
      rather than only `smoke`.
- [ ] The regenerated `results/tables/final_table_long.csv` has 1251 data rows and five distinct
      values in its `dataset` column — matching what `docs/full_run_2026-08-18.md` reports.
- [ ] `head -40 docs/full_run_2026-08-18.md` shows the errata block before section 1.
- [ ] `test ! -f slurm/full_pipeline_rerun.sbatch`
- [ ] `uv run pytest tests/ -q` still passes (no code changed, but the results-tree move can
      break a test that reads a fixture path).

---

## Commit

`[docs] errata on run 1 and repo hygiene before the rerun`
