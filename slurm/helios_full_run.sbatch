#!/bin/bash -l
#SBATCH --job-name=ctxai-run2
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.out
#SBATCH --error=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.err

# Submit once for each config, sequentially:
#   sbatch slurm/helios_full_run.sbatch full
#   sbatch --dependency=afterany:<full-job> slurm/helios_full_run.sbatch full_nl

set -uo pipefail

PROJECT=/net/home/plgrid/plgofurman/projects/causaltemp-xai
HEAVY=/net/storage/pr3/plgrid/plggcfsgenwro/users/plgofurman/causaltemp-xai
export XDG_CACHE_HOME="$HEAVY/cache/xdg"
export TORCH_HOME="$HEAVY/cache/torch"
export MPLCONFIGDIR="$HEAVY/cache/matplotlib"
export MPLBACKEND=Agg
module purge || exit 1
module load ML-bundle/25.10 || exit 1
source "$HEAVY/envs/gh200/bin/activate" || exit 1
cd "$PROJECT" || exit 1
mkdir -p slurm/logs

unset OMP_NUM_THREADS MKL_NUM_THREADS
CORES=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES

source "$PROJECT/slurm/helios_harness.sh"

if (( $# != 1 )) || [[ "$1" != "full" && "$1" != "full_nl" ]]; then
  echo "usage: sbatch $0 {full|full_nl}" >&2
  exit 2
fi
CONFIG=$1
N_CF=100
METHODS=(
  NoiselessSCMRecourse
  PearlSCMRecourse
  CftsWachter
  CftsCOMTE
  CftsConfeti
  CftsCels
  TSCausal
)

echo "=== PREFLIGHT config=$CONFIG job=${SLURM_JOB_ID:-none} $(date -Is) ==="
preflight_runtime || exit 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

prepare_clean_run() {
  local archive_root="$HEAVY/run2/pre_run2"
  mkdir -p "$archive_root/results"
  if [[ -d "results/$CONFIG" ]]; then
    if [[ ! -e "$archive_root/results/$CONFIG" ]]; then
      mv "results/$CONFIG" "$archive_root/results/$CONFIG"
    else
      mkdir -p "$archive_root/retries"
      mv "results/$CONFIG" "$archive_root/retries/${CONFIG}-${SLURM_JOB_ID:-$(date +%s)}"
    fi
  fi
  # Job A owns shared-table reset. Job B is chained after A and must retain A's
  # freshly appended rows.
  if [[ "$CONFIG" == "full" && -d results/tables ]]; then
    if [[ ! -e "$archive_root/tables" ]]; then
      mv results/tables "$archive_root/tables"
    else
      mkdir -p "$archive_root/retries"
      mv results/tables "$archive_root/retries/tables-${SLURM_JOB_ID:-$(date +%s)}"
    fi
  fi
  mkdir -p "results/$CONFIG" results/tables
}
prepare_clean_run || exit 1

step python experiments/01_generate_benchmarks.py --config "$CONFIG"
step python experiments/02_train_classifiers.py --config "$CONFIG"
step python experiments/03_run_cf_methods.py --config "$CONFIG" --n-cf "$N_CF" --methods "${METHODS[@]}"
step python experiments/04_evaluate_axes.py --config "$CONFIG"
step python experiments/05_run_oracle_control.py --config "$CONFIG" --n-cf "$N_CF"
step python experiments/07_auxiliary_methods.py --config "$CONFIG" --method pns --n-cf "$N_CF"

# Hard acceptance checks. A failed phase already increments FAILURES; these
# checks additionally reject missing or stale-schema output.
step test -s "results/$CONFIG/axis_a_benchmark.json"
step test -s "results/$CONFIG/lstm/train_report.json"
step test -s "results/$CONFIG/lstm/summary.json"
step test -s "results/$CONFIG/oracle/summary.json"
step test -s "results/$CONFIG/lstm/pns.json"
step python -c 'import pathlib,sys; p=pathlib.Path("results")/sys.argv[1]/"lstm"/"cf"; assert len(list(p.glob("X_cf_*.npy"))) == 7; assert len(list(p.glob("no_cf_found_*.npy"))) == 7' "$CONFIG"
step python -c 'import json,pathlib,sys; c=sys.argv[1]; d=json.loads((pathlib.Path("results")/c/"lstm"/"summary.json").read_text()); expected={"NoiselessSCMRecourse","PearlSCMRecourse","CftsWachter","CftsCOMTE","CftsConfeti","CftsCels","TSCausal"}; assert {m["method"] for m in d["methods"]} == expected; assert d["git_dirty"] is False' "$CONFIG"
step python -c 'import json,pathlib,sys; c=sys.argv[1]; d=json.loads((pathlib.Path("results")/c/"oracle"/"summary.json").read_text()); assert isinstance(d["methods"], list) and len(d["methods"]) == 2; p=next(m for m in d["methods"] if m["method"] == "OracleCF-Pearl"); assert p["do_complexity_mean_pearl_scorable"] == 1.0; assert all("validity" in m for m in d["methods"])' "$CONFIG"

finish
