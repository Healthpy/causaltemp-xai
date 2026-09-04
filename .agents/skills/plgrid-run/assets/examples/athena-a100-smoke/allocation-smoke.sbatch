#!/bin/bash -l
#SBATCH --job-name=a100-allocation-smoke
#SBATCH --account=plgtabcfs-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=allocation-smoke-%j.out
#SBATCH --error=allocation-smoke-%j.err

set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
GROUP="$PLG_GROUP"
PROJECT="$HOME/projects/plgrid-a100-smoke"
HEAVY="$PLG_GROUPS_STORAGE/$GROUP/users/$USER/plgrid-a100-smoke"

cd "$PROJECT"
test "$(uname -m)" = x86_64
test -n "${CUDA_VISIBLE_DEVICES:-}"
test -w "$HEAVY/outputs"
test -L "$PROJECT/.cache"
test -L "$PROJECT/.local"

printf 'job=%s\nhost=%s\narchitecture=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(uname -m)" "$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

RESULT="$HEAVY/outputs/allocation-$SLURM_JOB_ID.txt"
printf 'Athena A100 allocation and STORAGE-01 write smoke: PASS\n' | tee "$RESULT"
test -s "$RESULT"
