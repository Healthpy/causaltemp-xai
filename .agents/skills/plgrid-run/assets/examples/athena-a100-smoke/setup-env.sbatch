#!/bin/bash -l
#SBATCH --job-name=a100-env-setup
#SBATCH --account=plgtabcfs-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=setup-env-%j.out
#SBATCH --error=setup-env-%j.err

set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
GROUP="$PLG_GROUP"
PROJECT="$HOME/projects/plgrid-a100-smoke"
HEAVY="$PLG_GROUPS_STORAGE/$GROUP/users/$USER/plgrid-a100-smoke"
ENV_DIR="$HEAVY/envs/a100"

export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export HF_HUB_CACHE="$HEAVY/cache/huggingface-hub"
export HF_DATASETS_CACHE="$HEAVY/cache/huggingface-datasets"
export TORCH_HOME="$HEAVY/cache/torch"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0

test "$(uname -m)" = x86_64
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python -m venv --system-site-packages "$ENV_DIR"
elif ! grep -q '^include-system-site-packages = true$' "$ENV_DIR/pyvenv.cfg"; then
  python -m venv --upgrade --system-site-packages "$ENV_DIR"
fi

test -L "$PROJECT/.venv"
"$PROJECT/.venv/bin/python" -c \
  'import platform, torch; assert platform.machine() == "x86_64"; assert torch.cuda.is_available(); print(torch.__version__)'
