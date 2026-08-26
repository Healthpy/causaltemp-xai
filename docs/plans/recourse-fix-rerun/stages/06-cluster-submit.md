# Stage 6: Cluster preflight and submit

**Goal**: Verify the cluster is ready, then submit the chained jobs — **with explicit human
authorisation**.
**Dependencies**: Stage 5 fully green.

---

## MANDATORY STOP

Submitting spends roughly 26 GPU-hours of the `plgcountercontex-gpu-gh200` grant. Run every
preflight check below, print the exact `sbatch` commands and the full `sbatch --test-only`
output, then **stop and wait for explicit human authorisation**. Never submit unattended, and
never treat a previous session's approval as covering this one.

## Security constraints

- **Never read, print, echo or transfer `.env`.** It holds `PLG_PASSWORD`. Cluster access is
  key-based only. Read non-secret selection variables with a targeted grep that excludes the
  password key, e.g. `grep -E '^PLG_(LOGIN|GRANT)=' .env`.
- Every `rsync` to the cluster must carry `--exclude '.env'`.
- Do not use `--delete-excluded`: it reproducibly breaks macOS `openrsync`.

---

## Steps

1. **Check the grant balance.** `hpc-grants` on the login node. Confirm at least 26 GPU-hours
   remain on `plgcountercontex-gpu-gh200` after run 1's spend. If the balance is short, stop and
   report — do not submit a job that will be killed mid-run.

2. **Check the partition limits.** `sinfo -p plgrid-gpu-gh200 -o '%P %a %l %D %G'` and
   `scontrol show partition plgrid-gpu-gh200`. Confirm the `MaxTime` accommodates the walltimes
   stage 5 sized. This was an open question during planning and has never been measured.

3. **Check the environment survived.** Confirm `$HEAVY/envs/gh200` still exists and imports
   cleanly — submit `slurm/helios_import_check.sbatch` and wait for it. If the venv is gone,
   rebuild it with `slurm/helios_setup_env.sbatch` first; that must run **inside a GH200 job**,
   because the login nodes are `x86_64` and the compute nodes are `aarch64`.
   - **Never `uv sync`.** The committed `uv.lock` resolves torch 2.12.0 against CUDA 13; Helios
     needs torch 2.9.0+cu129 from `PIP_FIND_LINKS=/net/software/aarch64/el9/wheels/ML-bundle/25.10`.
     A wrong build is a silent CPU fallback, not an error.
   - `uv` is absent on both login and compute nodes. Use the venv's own pip.
   - `import causaltemp_xai` hard-requires `tigramite` via `causaltemp_xai/methods/causal/pcmci.py`,
     so `pip install --no-deps -e .` is not sufficient.

4. **Sync the code** with the tightened rsync from stage 4 (`--exclude '.env' --exclude 'results'
   --exclude 'notebooks'`), then verify on the cluster that `.env` is absent and that
   `git status --porcelain -- ':!results' ':!notebooks' ':!slurm/logs'` is clean. This is the
   exact source-clean pathspec used by `experiments/_common.py::git_provenance()`.

5. **Validate the job scripts without consuming allocation.** Run and record:
   ```bash
   sbatch --test-only slurm/helios_full_run.sbatch full
   sbatch --test-only slurm/helios_full_run.sbatch full_nl
   sbatch --test-only slurm/helios_reports.sbatch
   ```

6. **Run the smoke job first.** Submit `slurm/helios_smoke.sbatch` and confirm it completes
   `COMPLETED` with `ExitCode 0:0`, contains no `!!! FAILED` marker, and passes every hard
   acceptance check before submitting the expensive chain. The Stage-4 wrapper must return
   non-zero if any phase failed; do not accept stale output files as evidence of this run.

7. **STOP. Present to the human**: the grant balance, the partition `MaxTime`, the smoke job's
   result, the exact `sbatch` commands, and the `--test-only` output. Wait for a clear yes.

8. **Submit the chain** once authorised:
   - Job A: `full`, `n_cf=100`
   - Job B: `full_nl`, `n_cf=100`, `--dependency=afterany:<A>`
   - Job C: reports and the final table, `--dependency=afterany:<B>`
   - Use `--dependency=afterany`, not `afterok`: a partial failure in A should still let B run,
     because `_common.py::append_table`'s race is the thing being avoided, not A's success.
   - Capture every job ID with `sbatch --parsable` and write them to
     `$HEAVY/run2/run2.jobids`, never inside the Git checkout. Record the same IDs in the index.

9. **Report the job IDs and stop monitoring.** Do not poll. Confirm submission with a single
   `squeue -j` call and leave the jobs to run.

---

## Verification

- [ ] `hpc-grants` shows sufficient GPU-hours; the number is recorded in the index.
- [ ] The partition `MaxTime` is recorded and accommodates the requested walltimes.
- [ ] `slurm/helios_import_check.sbatch` completed `0:0`.
- [ ] `.env` is confirmed absent on the cluster.
- [ ] `git status --porcelain -- ':!results' ':!notebooks'` is clean on the cluster.
- [ ] `sbatch --test-only` passes for every job.
- [ ] The smoke job completed `COMPLETED` / `0:0`.
- [ ] **Explicit human authorisation was given** and is recorded in the index.
- [ ] Every job ID is recorded in the index and in `$HEAVY/run2/run2.jobids`; no
      `slurm/run2.jobids` exists.

---

## Commit

`[slurm] submit the run-2 Helios chain`
