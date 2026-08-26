#!/bin/bash -l
#SBATCH --job-name=ctxai-imports
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.out
#SBATCH --error=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.err

# Import-closure check for the GH200 env built by slurm/helios_setup_env.sbatch.
# Cannot run on the login node: that is x86_64, the env is aarch64.
#
# Runs the real import path (experiments/01_generate_benchmarks.py --config
# smoke: k=5, T=30, N=500) and repairs genuinely-missing modules in place,
# with torch pinned by a constraints file so nothing can pull a different
# build over the verified torch==2.9.0+cu129.

set -uo pipefail

PROJECT=/net/home/plgrid/plgofurman/projects/causaltemp-xai
HEAVY=/net/storage/pr3/plgrid/plggcfsgenwro/users/plgofurman/causaltemp-xai
ENV_DIR="$HEAVY/envs/gh200"

mkdir -p "$PROJECT/slurm/logs"

export XDG_CACHE_HOME="$HEAVY/cache/xdg"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export TORCH_HOME="$HEAVY/cache/torch"
export MPLCONFIGDIR="$HEAVY/cache/matplotlib"
export MPLBACKEND=Agg
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR"

module purge || exit 1
module load ML-bundle/25.10 || exit 1
unset OMP_NUM_THREADS MKL_NUM_THREADS
CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES
test "$(uname -m)" = aarch64 || exit 1

source "$ENV_DIR/bin/activate" || exit 1
cd "$PROJECT" || exit 1
source "$PROJECT/slurm/helios_harness.sh"
preflight_source_clean || exit 1

echo "=== context ==="
hostname
python -V
which python
nvidia-smi --query-gpu=name,driver_version --format=csv || true

if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then export UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL"; fi
if [[ -n "${PIP_FIND_LINKS:-}" ]]; then export UV_FIND_LINKS="$PIP_FIND_LINKS"; fi

CONSTRAINTS="$HEAVY/cache/helios-constraints.txt"
echo 'torch==2.9.0+cu129' > "$CONSTRAINTS"

if command -v uv >/dev/null 2>&1; then
  INSTALL=(uv pip install --python "$ENV_DIR/bin/python" -c "$CONSTRAINTS")
else
  INSTALL=(python -m pip install -c "$CONSTRAINTS")
fi

# python module name -> pip distribution name
pip_name_for() {
  case "$1" in
    sklearn)     echo scikit-learn ;;
    yaml)        echo PyYAML ;;
    cv2)         echo opencv-python-headless ;;
    PIL)         echo pillow ;;
    dateutil)    echo python-dateutil ;;
    pkg_resources) echo setuptools ;;
    *)           echo "$1" ;;
  esac
}

# Run a python snippet; on ModuleNotFoundError install the missing dist and retry.
run_with_repair() {
  local label="$1" snippet="$2" max="${3:-6}"
  local attempt=0 out rc missing top pkg
  while :; do
    attempt=$((attempt + 1))
    out="$(python -c "$snippet" 2>&1)"
    rc=$?
    echo "--- [$label] attempt $attempt (rc=$rc)"
    printf '%s\n' "$out"
    if [[ $rc -eq 0 ]]; then
      return 0
    fi
    if [[ $attempt -ge $max ]]; then
      echo "!!! [$label] giving up after $attempt attempts"
      return $rc
    fi
    missing="$(printf '%s\n' "$out" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)"
    if [[ -z "$missing" ]]; then
      echo "!!! [$label] failure is not a missing module -- not auto-repairable"
      return $rc
    fi
    top="${missing%%.*}"
    pkg="$(pip_name_for "$top")"
    echo ">>> [$label] missing module '$top' -> installing pip package '$pkg'"
    if ! "${INSTALL[@]}" "$pkg"; then
      echo "!!! [$label] install of '$pkg' failed"
      return 1
    fi
  done
}

echo "=== 1. core import closure ==="
step run_with_repair core \
  'import causaltemp_xai, numpy, scipy, sklearn, networkx, pandas, statsmodels, matplotlib, seaborn, torch; print("core ok")'

echo "=== 2. experiments package ==="
step python -c "import experiments; print('experiments ok')"

echo "=== 3. cfts (third_party/cfts_repo) -- first look, non-fatal ==="
python -c "import sys; sys.path.insert(0,'third_party/cfts_repo'); import cfts; print('cfts ok')" || \
  echo "cfts missing before repair; final preflight will require it"

echo "=== 4. real import path: 01_generate_benchmarks.py --config smoke ==="
step run_with_repair smoke-gen \
  'import runpy, sys; sys.argv=["experiments/01_generate_benchmarks.py","--config","smoke"]; runpy.run_path("experiments/01_generate_benchmarks.py", run_name="__main__")'

echo "=== 5. re-run the phase as a plain subprocess (exact CLI) ==="
step python experiments/01_generate_benchmarks.py --config smoke

echo "=== 6. best-effort: cfts optional deps (non-fatal) ==="
"${INSTALL[@]}" captum tqdm stumpy pyts tslearn dtaidistance fastdtw || \
  echo "optional cfts dependency install failed; smoke will exercise the live methods"
step python -c "import sys; sys.path.insert(0,'third_party/cfts_repo'); import cfts; print('cfts ok')"

echo "=== 7. final environment re-verification ==="
step python - <<'PY'
import platform
import sys

import torch

assert platform.machine() == "aarch64", platform.machine()
print("machine  :", platform.machine())
print("python   :", sys.version.split()[0])
print("torch    :", torch.__version__)
print("cuda ver :", torch.version.cuda)
print("cuda avail:", torch.cuda.is_available())
assert torch.__version__.startswith("2.9.0+cu129"), f"torch was clobbered: {torch.__version__}"
assert torch.cuda.is_available(), "CUDA disappeared"
print("gpu      :", torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
print("gpu matmul sum:", float((x @ x).sum()))

import importlib

for m in ("numpy", "scipy", "sklearn", "networkx", "pandas", "statsmodels", "matplotlib", "seaborn"):
    try:
        print(f"{m:14s}", importlib.import_module(m).__version__)
    except Exception as exc:  # noqa: BLE001
        print(f"{m:14s} FAILED {exc!r}")
PY

echo "=== 8. generated dataset ==="
ls -la data/scm_t/smoke 2>/dev/null || true
ls -la results/smoke 2>/dev/null || true

step preflight_runtime
finish
