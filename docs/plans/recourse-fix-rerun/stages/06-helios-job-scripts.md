# Stage 6: Helios job scripts

**Goal**: Turn the run-1 prose recipe into committed, `--test-only`-validated sbatch files that
carry the threading fix and fail loudly at preflight.
**Dependencies**: Stages 3-5 (the scripts should pin the code state they will run).

---

## Why this stage exists

The run-1 recipe exists only as prose in `docs/full_run_2026-08-18.md` section 3. No file under
`slurm/` reproduces it. And `grep -n "OMP_NUM_THREADS\|MKL_NUM_THREADS\|SLURM_CPUS_PER_TASK"
slurm/*.sbatch` returns nothing — so resubmitting `slurm/helios_smoke.sbatch` today inherits
`OMP_NUM_THREADS=1` from `module load ML-bundle/25.10` and runs single-threaded on a 16-core
allocation. In run 1 that mistake cost a resubmission and would have produced a TIMEOUT after
about 16 wasted GPU-hours.

---

## Steps

1. **Write `slurm/helios_full_run.sbatch`.** One script, parameterised by config and submitted
   twice (`full`, then `full_nl`) as the first two jobs of an `afterany` chain. Include, in order:
   - The `#SBATCH` block matching run 1's measured resources: `--nodes=1 --ntasks-per-node=1
     --cpus-per-task=16 --mem=24G --gres=gpu:1`, partition `plgrid-gpu-gh200`, account
     `plgcountercontex-gpu-gh200`. Peak MaxRSS in run 1 was about 3 GB, so 24G is ample — do not
     over-request.
   - Walltime from run 1's measured elapsed plus headroom: `full` took 12:07:25 under a 16:00:00
     budget, `full_nl` took 12:05:35 under 30:00:00. **Add headroom for the lambda backoff from
     stage 3** — it can multiply optimiser steps on hard instances. Use stage 7's measured
     smoke-tier delta to size it.
   - The mandatory preamble from `resources/commands.md` (project and heavy paths, cache
     variables, `module purge`, `module load ML-bundle/25.10`, venv activation).
   - **The threading fix, verbatim:**
     ```bash
     unset OMP_NUM_THREADS MKL_NUM_THREADS
     CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
     export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES
     ```
     Never `export OMP_NUM_THREADS=$(nproc)` — GNU `nproc` obeys `OMP_NUM_THREADS`, so after the
     module load that reads back 1 and re-exports 1.
   - **Preflight gates that exit non-zero**, before any real work: `torch.cuda.is_available()`;
     the expected device name; `torch.get_num_threads() >= 2`; `import cfts` (a silent failure
     here loses 5 of the 7 methods and produces a quietly wrong table); and a git-cleanliness
     check that uses the same code-path exclusions as `git_provenance()`.
   - `set -uo pipefail` with a failure-accumulating `step()` wrapper rather than `set -e`, so a
     late phase failing does not discard earlier results but the job still fails at the end:
     ```bash
     FAILURES=0
     step() {
       "$@"
       rc=$?
       if (( rc != 0 )); then FAILURES=$((FAILURES + 1)); fi
       return 0
     }
     # Run all phases and hard acceptance checks, then:
     if (( FAILURES != 0 )); then exit 1; fi
     ```
     Every expected output check increments `FAILURES`; it must not merely print `MISSING`.
   - The phase sequence from `resources/commands.md`.

2. **Add the threading fix, matching code-clean preflight, and accumulated final status to the
   existing scripts.**
   `slurm/helios_smoke.sbatch`, `slurm/helios_import_check.sbatch`,
   `slurm/helios_setup_env.sbatch` all lack at least one of them. Setup/import checks may still
   fail immediately where continuing would be meaningless; the smoke script must run all
   diagnostics and then exit non-zero if any phase or acceptance check failed.

3. **Write `slurm/helios_reports.sbatch`.** Give it the same module/venv/threading/code-clean
   preflight and accumulated-failure contract. It runs only the Job-C sequence from
   `resources/commands.md`: figures, both PNS reports, and
   `make_final_table.py --configs full full_nl`. Give it a measured reports walltime plus
   headroom; it must not accept a config argument. Add hard checks that the final long table and
   provenance JSON exist, are non-empty, and name exactly `full` and `full_nl`.

4. **Write the submission driver** — a short shell script or a documented command block that
   submits the three jobs with `--dependency=afterany` and records every returned job ID to a
   file. Sequential execution is mandatory: `experiments/_common.py::append_table` does an
   unlocked read-modify-write on the shared cross-run CSV, before `summary.json` is written.
   Two concurrent configs corrupt it.

   Store the ID file outside the Git checkout at `$HEAVY/run2/run2.jobids` (create the directory
   first), then copy the IDs into the plan index after submission. Never create
   `slurm/run2.jobids`; it would fail Job B's cleanliness preflight.

5. **Define code-clean provenance and tighten the code sync.** Run 1 recorded `git_dirty: true` because the rsync skipped
   `results/` and `notebooks/` — both tracked — so git on the cluster saw them as deleted. No
   `.py` file differed. Make that explicit: an `rsync` with `--exclude '.env' --exclude
   'results' --exclude 'notebooks'` (plus the usual heavy excludes), then a preflight
   `git status --porcelain -- ':!results' ':!notebooks' ':!slurm/logs'` that fails the job on any
   other dirt.
   - In `experiments/_common.py::git_provenance()`, define `git_dirty` as **source/code dirt**
     using these same pathspec exclusions, and add `git_worktree_dirty` from unfiltered status as
     a separate diagnostic. Generated or deliberately omitted result trees may make
     `git_worktree_dirty=true`; they must not make a verified committed code revision report
     `git_dirty=true`.
   - In `experiments/make_final_table.py`, propagate both fields into per-config and final-table
     provenance; do not collapse them into one Boolean.
   - Add a unit test that dirt under `results/` leaves `git_dirty=false`, while a modified `.py`
     file makes it true. Clear the function's LRU cache between cases.
   - `--delete-excluded` is known to break macOS `openrsync`. Do not use it.
   - **`.env` must never be transferred.** It holds `PLG_PASSWORD`. Access is key-based only.

---

## Verification

- [ ] `bash -n slurm/*.sbatch` — every script parses.
- [ ] `test -f slurm/helios_reports.sbatch`, and its body contains all four Job-C commands.
- [ ] `grep -L 'SLURM_CPUS_PER_TASK' slurm/helios_*.sbatch` returns nothing: every Helios script
      carries the threading fix.
- [ ] `grep -rn 'OMP_NUM_THREADS=\$(nproc)' slurm/` returns nothing.
- [ ] `grep -rn 'uv sync\|uv run' slurm/` returns nothing.
- [ ] `grep -n "exclude '.env'\|exclude .env" ` finds the exclusion in the sync command.
- [ ] The scripts name account `plgcountercontex-gpu-gh200` and partition `plgrid-gpu-gh200`,
      and no script references `tesr123566` or `gpu_a100`.
- [ ] No command writes `slurm/run2.jobids`; the driver writes `$HEAVY/run2/run2.jobids`.
- [ ] A harness test forces one `step()` failure and confirms the script's final status is
      non-zero after later steps still run.
- [ ] Provenance tests distinguish `git_dirty` (source paths) from `git_worktree_dirty` (all paths).
- [ ] `sbatch --test-only` is deferred to stage 8 (it needs the cluster). Note that here.

---

## Commit

`[slurm] add fail-safe Helios run and reports jobs with code-clean provenance`
