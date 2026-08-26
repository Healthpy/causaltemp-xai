#!/bin/bash -l
#SBATCH --job-name=ctxai-smoke
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.out
#SBATCH --error=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.err

# End-to-end pre-submission smoke. All phases and acceptance checks run; any
# failure makes the final job status non-zero.

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

if (( $# != 0 )); then
  echo "usage: sbatch $0" >&2
  exit 2
fi

CONFIG=smoke
N_CF=5
METHODS=(
  NoiselessSCMRecourse
  PearlSCMRecourse
  CftsWachter
  CftsCOMTE
  CftsConfeti
  CftsCels
  TSCausal
)

echo "=== PREFLIGHT smoke job=${SLURM_JOB_ID:-none} $(date -Is) ==="
preflight_runtime || exit 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# Keep historical smoke outputs, but never let old method arrays contaminate
# the canonical seven-method acceptance checks below.
archive_root="$HEAVY/run2/pre_run2"
mkdir -p "$archive_root/results"
if [[ -d "results/$CONFIG" ]]; then
  if [[ ! -e "$archive_root/results/$CONFIG" ]]; then
    mv "results/$CONFIG" "$archive_root/results/$CONFIG"
  else
    mkdir -p "$archive_root/retries"
    mv "results/$CONFIG" "$archive_root/retries/${CONFIG}-${SLURM_JOB_ID:-$(date +%s)}"
  fi
fi
mkdir -p "results/$CONFIG"

step python experiments/01_generate_benchmarks.py --config "$CONFIG"
step python experiments/02_train_classifiers.py --config "$CONFIG"
step python experiments/03_run_cf_methods.py --config "$CONFIG" --n-cf "$N_CF" --methods "${METHODS[@]}"
step python experiments/04_evaluate_axes.py --config "$CONFIG"
step python experiments/05_run_oracle_control.py --config "$CONFIG" --n-cf "$N_CF"
step python experiments/07_auxiliary_methods.py --config "$CONFIG" --method pns --n-cf "$N_CF"
step python experiments/08_aggregate_and_report.py figures
step python experiments/08_aggregate_and_report.py pns --config "$CONFIG" --seeds
step python experiments/make_final_table.py --configs "$CONFIG"

step test -s "results/$CONFIG/lstm/summary.json"
step test -s "results/$CONFIG/oracle/summary.json"
step test -s "results/$CONFIG/lstm/pns.json"
step test -s results/tables/final_table_long.csv
step test -s results/tables/final_table_provenance.json
step python -c 'import pathlib,sys; p=pathlib.Path("results")/sys.argv[1]/"lstm"/"cf"; assert len(list(p.glob("X_cf_*.npy"))) == 7; assert len(list(p.glob("no_cf_found_*.npy"))) == 7' "$CONFIG"
step python -c 'import json,pathlib,sys; c=sys.argv[1]; d=json.loads((pathlib.Path("results")/c/"lstm"/"summary.json").read_text()); expected={"NoiselessSCMRecourse","PearlSCMRecourse","CftsWachter","CftsCOMTE","CftsConfeti","CftsCels","TSCausal"}; assert {m["method"] for m in d["methods"]} == expected; assert d["git_dirty"] is False' "$CONFIG"
step python -c 'import pandas as pd,sys; d=pd.read_csv("results/tables/final_table_long.csv"); assert set(d["dataset"]) == {sys.argv[1]}' "$CONFIG"

finish
