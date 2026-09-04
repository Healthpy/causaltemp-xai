#!/usr/bin/env bash
set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"

cd "$HOME/projects/plgrid-gh200-smoke"

sbatch --test-only allocation-smoke.sbatch

allocation_job="$(sbatch --parsable allocation-smoke.sbatch)"
setup_job="$(sbatch --parsable --dependency="afterok:$allocation_job" setup-env.sbatch)"
torch_job="$(sbatch --parsable --dependency="afterok:$setup_job" torch-smoke.sbatch)"

printf '%s\n' "$allocation_job" > allocation.jobid
printf '%s\n' "$setup_job" > setup.jobid
printf '%s\n' "$torch_job" > torch.jobid

printf 'allocation_job=%s\nsetup_job=%s\ntorch_job=%s\n' \
  "$allocation_job" "$setup_job" "$torch_job"
squeue -j "$allocation_job,$setup_job,$torch_job"
