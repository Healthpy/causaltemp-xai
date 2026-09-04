#!/bin/bash -l
#SBATCH --job-name=gh200-env-setup
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=setup-env-%j.out
#SBATCH --error=setup-env-%j.err

set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
GROUP="$PLG_GROUP"
PROJECT="$HOME/projects/plgrid-gh200-smoke"
HEAVY="$PLG_GROUPS_STORAGE/$GROUP/users/$USER/plgrid-gh200-smoke"
ENV_DIR="$HEAVY/envs/gh200"

export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export HF_HUB_CACHE="$HEAVY/cache/huggingface-hub"
export HF_DATASETS_CACHE="$HEAVY/cache/huggingface-datasets"
export TORCH_HOME="$HEAVY/cache/torch"

module purge
module load ML-bundle/25.10

test "$(uname -m)" = aarch64
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python -m venv "$ENV_DIR"
fi

if command -v uv >/dev/null 2>&1; then
  if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    export UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL"
  fi
  uv pip install --python "$ENV_DIR/bin/python" 'torch==2.9.0+cu129'
else
  "$ENV_DIR/bin/python" -m pip install 'torch==2.9.0+cu129'
fi

test -L "$PROJECT/.venv"
"$PROJECT/.venv/bin/python" -c \
  'import platform, torch; assert platform.machine() == "aarch64"; print(torch.__version__)'
