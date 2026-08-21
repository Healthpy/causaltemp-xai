#!/bin/bash
#SBATCH --job-name=ctxai-full-rerun
#SBATCH --account=tesr123566
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Regenerates results/full and results/full_nl on current metric code
# (post-99a2e9f construct-validity fixes) to close the M1 blocker recorded
# in DECISIONS.md 2026-07-23: docs/hypotheses_assessment.md verdicts are
# provisional until this re-run lands. Uses the locked paper configs (FULL,
# FULL_NL; causaltemp_xai/config.py) and the locked n-cf=100 (README.md,
# matches the row count of the previous results/full/lstm/per_instance.csv).
#
# GPU partition -- this account's compute budget is GPU-hours only (a CPU
# partition submission is rejected outright: "No compute budget was found
# for the partition rome"). Job 24833327 originally failed on this same
# partition after ~10min: NoiselessSCMRecourse/PearlSCMRecourse hit a device-placement bug and
# every gradient-based Cfts method plus integrated_gradients hit "cudnn RNN
# backward can only be called in training mode" (computing gradients through
# the LSTM in .eval() mode under cuDNN). Both are now fixed in
# causaltemp_xai/methods/counterfactual/scm_recourse.py and
# causaltemp_xai/methods/attribution/integrated_gradients.py /
# causaltemp_xai/methods/counterfactual/cfts_methods.py -- see those diffs.
#
# Submit with: sbatch slurm/full_pipeline_rerun.sbatch
# Monitor with: squeue -u $USER ; tail -f slurm/logs/ctxai-full-rerun-<jobid>.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs

run() { echo "+++ $*"; uv run --no-sync "$@"; }

N_CF=100

echo "=== git state ==="
git rev-parse HEAD
git status --porcelain | head -5 || true

echo "=== FULL (linear) ==="
run python experiments/01_generate_benchmarks.py --config full
run python experiments/02_train_classifiers.py --config full
run python experiments/03_run_cf_methods.py --config full --n-cf "$N_CF"
run python experiments/04_evaluate_axes.py --config full
run python experiments/05_run_oracle_control.py --config full

echo "=== FULL_NL (nonlinear) ==="
run python experiments/01_generate_benchmarks.py --config full_nl
run python experiments/02_train_classifiers.py --config full_nl
run python experiments/03_run_cf_methods.py --config full_nl --n-cf "$N_CF"
run python experiments/04_evaluate_axes.py --config full_nl
run python experiments/05_run_oracle_control.py --config full_nl
run python experiments/07_auxiliary_methods.py --config full_nl --method dynotears

echo "=== Aggregation ==="
run python experiments/08_aggregate_and_report.py figures

echo "=== Done ==="
git status --porcelain -- results/ | head -20 || true
