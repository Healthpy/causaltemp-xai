#!/bin/bash -l
#SBATCH --job-name=a100-torch-smoke
#SBATCH --account=plgtabcfs-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=torch-smoke-%j.out
#SBATCH --error=torch-smoke-%j.err

set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
GROUP="$PLG_GROUP"
PROJECT="$HOME/projects/plgrid-a100-smoke"
HEAVY="$PLG_GROUPS_STORAGE/$GROUP/users/$USER/plgrid-a100-smoke"

export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export HF_HUB_CACHE="$HEAVY/cache/huggingface-hub"
export HF_DATASETS_CACHE="$HEAVY/cache/huggingface-datasets"
export TORCH_HOME="$HEAVY/cache/torch"
export OMP_NUM_THREADS=1

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0
cd "$PROJECT"

srun "$PROJECT/.venv/bin/python" torch_smoke.py \
  --output "$HEAVY/outputs/torch-$SLURM_JOB_ID.json"
