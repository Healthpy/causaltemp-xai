#!/bin/bash -l
#SBATCH --job-name=ctxai-reports
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.out
#SBATCH --error=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.err

# Job C. It intentionally accepts no config argument and publishes only full + full_nl.

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

echo "=== PREFLIGHT reports job=${SLURM_JOB_ID:-none} $(date -Is) ==="
preflight_runtime || exit 1

archive_root="$HEAVY/run2/pre_run2"
mkdir -p "$archive_root"
if [[ -d results/figures ]]; then
  if [[ ! -e "$archive_root/figures" ]]; then
    mv results/figures "$archive_root/figures"
  else
    mkdir -p "$archive_root/retries"
    mv results/figures "$archive_root/retries/figures-${SLURM_JOB_ID:-$(date +%s)}"
  fi
fi
mkdir -p results/figures results/tables

step python experiments/08_aggregate_and_report.py figures
step python experiments/08_aggregate_and_report.py pns --config full --seeds
step python experiments/08_aggregate_and_report.py pns --config full_nl --seeds
step python experiments/make_final_table.py --configs full full_nl

LONG=results/tables/final_table_long.csv
PROVENANCE=results/tables/final_table_provenance.json
step test -s "$LONG"
step test -s "$PROVENANCE"
step python -c 'import pandas as pd; d=pd.read_csv("results/tables/final_table_long.csv"); assert set(d["dataset"]) == {"full", "full_nl"}; assert len(d) > 0'
step python -c 'import json; d=json.load(open("results/tables/final_table_provenance.json")); assert set(d) - {"_suppressed_metrics"} == {"full", "full_nl"}; assert all(d[c][k]["git_dirty"] is False for c in ("full","full_nl") for k in d[c])'

finish
