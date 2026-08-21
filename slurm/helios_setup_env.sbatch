#!/bin/bash -l
#SBATCH --job-name=ctxai-env
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.out
#SBATCH --error=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.err

# Build the aarch64 / GH200 Python environment for causaltemp-xai.
#
# Deliberately does NOT apply the project lock: it resolves
# torch==2.12.0 against CUDA 13 (nvidia-*-cu13), which the Helios driver
# stack can reject and silently fall back to CPU. The verified combination
# on Helios is ML-bundle/25.10 (CUDA 12.9.1) + torch==2.9.0+cu129, which is
# present as a prebuilt aarch64 wheel in the module's PIP_FIND_LINKS
# wheelhouse.
#
# The repo package is installed with --no-deps on purpose: its declared
# dependencies drag in tigramite / timeshap / captum / shap==0.42.1, and
# shap 0.42.1 has no aarch64 wheel for the module's Python (3.13).
#
# Submit with: sbatch slurm/helios_setup_env.sbatch

set -uo pipefail

PROJECT=/net/home/plgrid/plgofurman/projects/causaltemp-xai
HEAVY=/net/storage/pr3/plgrid/plggcfsgenwro/users/plgofurman/causaltemp-xai
ENV_DIR="$HEAVY/envs/gh200"

mkdir -p "$PROJECT/slurm/logs"

# --- caches: keep everything off the 10 GB $HOME quota -----------------------
export XDG_CACHE_HOME="$HEAVY/cache/xdg"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export TORCH_HOME="$HEAVY/cache/torch"
export MPLCONFIGDIR="$HEAVY/cache/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR"

module purge || exit 1
module load ML-bundle/25.10 || exit 1
unset OMP_NUM_THREADS MKL_NUM_THREADS
CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES
cd "$PROJECT" || exit 1
source "$PROJECT/slurm/helios_harness.sh"
preflight_source_clean || exit 1

echo "=== node context ==="
hostname
uname -m
nvidia-smi || true
echo "PIP_FIND_LINKS=${PIP_FIND_LINKS:-<unset>}"
echo "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-<unset>}"
echo "uv: $(command -v uv || echo '<absent>')"
ls -1 /usr/bin/python3* 2>/dev/null || true

test "$(uname -m)" = aarch64 || exit 1

# --- pick a bootstrap interpreter -------------------------------------------
# The module wheelhouse ships torch as cp313 wheels, so prefer Python 3.13.
PY_BOOT=""
for cand in python3.13 python3 python; do
  p="$(command -v "$cand" 2>/dev/null || true)"
  if [[ -z "$p" ]]; then continue; fi
  v="$("$p" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
  echo "python candidate: $p -> $v"
  if [[ "$v" == "3.13" ]]; then
    PY_BOOT="$p"
    break
  fi
  if [[ -z "$PY_BOOT" ]]; then
    PY_BOOT="$p"
  fi
done
if [[ -z "$PY_BOOT" ]]; then
  echo "FATAL: no python interpreter found after 'module load ML-bundle/25.10'" >&2
  exit 2
fi
echo "bootstrap python: $PY_BOOT ($("$PY_BOOT" -V 2>&1))"

# --- venv --------------------------------------------------------------------
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  mkdir -p "$(dirname "$ENV_DIR")"
  "$PY_BOOT" -m venv "$ENV_DIR"
fi
VPY="$ENV_DIR/bin/python"
echo "venv python: $VPY ($("$VPY" -V 2>&1))"

step "$VPY" -m pip install --upgrade pip setuptools wheel

# --- installer ---------------------------------------------------------------
if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
  export UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL"
fi
if [[ -n "${PIP_FIND_LINKS:-}" ]]; then
  export UV_FIND_LINKS="$PIP_FIND_LINKS"
fi

if command -v uv >/dev/null 2>&1; then
  echo "installer: uv pip ($(uv --version))"
  INSTALL=(uv pip install --python "$VPY")
else
  echo "installer: venv pip"
  INSTALL=("$VPY" -m pip install)
fi

# 1. torch first, so nothing else gets to pick a torch build for us.
echo "=== install torch==2.9.0+cu129 ==="
step "${INSTALL[@]}" 'torch==2.9.0+cu129'

# 2. runtime deps actually imported by the phases we run.
echo "=== install runtime deps ==="
step "${INSTALL[@]}" numpy scipy scikit-learn networkx pandas statsmodels matplotlib seaborn

# 2b. tigramite is NOT optional despite --no-deps below: causaltemp_xai/__init__.py
#     -> methods/__init__.py -> methods/causal/pcmci.py imports it at module level,
#     so plain `import causaltemp_xai` fails without it. It is pure Python
#     (numpy/scipy/six only) and installs cleanly on aarch64/py3.13.
#     captum is imported lazily by methods/counterfactual/cfts_methods.py and by
#     several third_party/cfts_repo algorithms; cheap and aarch64-clean, so take it.
echo "=== install tigramite + captum ==="
step "${INSTALL[@]}" tigramite captum

# 2c. optional: third_party/cfts_repo degrades gracefully without these (it only
#     warns), but the Cfts methods used in phase 03 need them at call time.
echo "=== install cfts extras (non-fatal) ==="
step "${INSTALL[@]}" tqdm stumpy pyts tslearn dtaidistance fastdtw

# 3. repo package itself, without its declared dependencies.
echo "=== install causaltemp-xai (--no-deps -e) ==="
install_repo() {
  if "${INSTALL[@]}" --no-deps -e "$PROJECT"; then
    return 0
  fi
  echo "editable install failed; retrying without build isolation"
  "${INSTALL[@]}" --no-deps --no-build-isolation -e "$PROJECT"
}
step install_repo

# --- verification ------------------------------------------------------------
echo "=== verification ==="
step "$VPY" - <<'PY'
import platform
import sys

import torch

ok = True
print("platform.machine()      :", platform.machine())
print("python                  :", sys.version.replace("\n", " "))
print("torch.__version__       :", torch.__version__)
print("torch.version.cuda      :", torch.version.cuda)
print("torch.cuda.is_available :", torch.cuda.is_available())

if platform.machine() != "aarch64":
    print("FATAL: not aarch64", file=sys.stderr)
    ok = False

if not torch.cuda.is_available():
    print("FATAL: CUDA not available -- refusing to accept a CPU fallback", file=sys.stderr)
    ok = False
else:
    print("torch.cuda.device_count :", torch.cuda.device_count())
    print("get_device_name(0)      :", torch.cuda.get_device_name(0))
    print("capability(0)           :", torch.cuda.get_device_capability(0))
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul on GPU           :", tuple(c.shape), c.device, float(c.float().sum()))

if not ok:
    raise SystemExit(1)
print("ENV OK")
PY

echo "=== installed versions ==="
step "$VPY" - <<'PY'
import importlib

for mod, attr in [
    ("torch", "__version__"),
    ("numpy", "__version__"),
    ("scipy", "__version__"),
    ("sklearn", "__version__"),
    ("networkx", "__version__"),
    ("pandas", "__version__"),
    ("statsmodels", "__version__"),
    ("matplotlib", "__version__"),
    ("seaborn", "__version__"),
]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, attr, '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{mod:14s} IMPORT FAILED: {exc!r}")
PY

step source "$ENV_DIR/bin/activate"
step preflight_runtime
finish
