#!/bin/bash -l
#SBATCH --job-name=gh200-torch-smoke
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=torch-smoke-%j.out
#SBATCH --error=torch-smoke-%j.err

set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
GROUP="$PLG_GROUP"
PROJECT="$HOME/projects/plgrid-gh200-smoke"
HEAVY="$PLG_GROUPS_STORAGE/$GROUP/users/$USER/plgrid-gh200-smoke"

export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export HF_HUB_CACHE="$HEAVY/cache/huggingface-hub"
export HF_DATASETS_CACHE="$HEAVY/cache/huggingface-datasets"
export TORCH_HOME="$HEAVY/cache/torch"

module purge
module load ML-bundle/25.10
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT"

srun python torch_smoke.py --output "$HEAVY/outputs/torch-$SLURM_JOB_ID.json"
