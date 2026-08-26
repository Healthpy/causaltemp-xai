#!/bin/bash -l
#SBATCH --job-name=ctxai-full
#SBATCH --account=tesei19978
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Snellius (SURF) A100 entry point -- the Helios/GH200 scripts next to this
# one target a different cluster, a different account and another user's
# paths, so they cannot be reused here.
#
# Runs Phase 03 (CF generation, every registered method) followed by Phase 04
# (Axis-C + CF-faith metrics) for one config:
#
#   sbatch slurm/snellius_full_run.sbatch full
#   sbatch slurm/snellius_full_run.sbatch full_nl
#
# One GPU per job, so the two configs run in parallel on separate GPUs.
# gpu_a100 bills a quarter node (18 of 72 cores) per GPU.

set -uo pipefail

CONFIG=${1:?usage: sbatch slurm/snellius_full_run.sbatch CONFIG [N_CF] [PHASES]}
N_CF=${2:-100}
# Space-separated experiment scripts to run, in order. Phase 03 regenerates
# the CFs *and* the Axis-B Shift-VR pass (a second full generation on the
# shifted test set); pass "04_evaluate_axes" alone to score CF arrays that
# are already on disk.
PHASES=${3:-"03_run_cf_methods 04_evaluate_axes"}

PROJECT=${SLURM_SUBMIT_DIR:-/gpfs/home6/echukwu/causaltemp-xai}
cd "$PROJECT" || exit 1
mkdir -p slurm/logs

PY="$PROJECT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv interpreter at $PY" >&2; exit 1; }

CORES=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "=== node=$(hostname) config=$CONFIG n_cf=$N_CF phases=[$PHASES] cores=$CORES start=$(date -Is)"
echo "=== git: $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) files"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

rc=0
for phase in $PHASES; do
  args=(--config "$CONFIG")
  [[ $phase == 03_run_cf_methods ]] && args+=(--n-cf "$N_CF")
  echo; echo "+++ $phase $(date -Is)"
  t0=$(date +%s)
  "$PY" "experiments/${phase}.py" "${args[@]}"
  prc=$?
  echo "+++ $phase rc=$prc seconds=$(( $(date +%s) - t0 ))"
  (( prc != 0 )) && { rc=$prc; break; }
done

echo "=== end=$(date -Is) rc=$rc"
exit $rc
