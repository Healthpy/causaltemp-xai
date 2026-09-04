#!/bin/bash -l
#SBATCH --job-name=ctxai-retired
#SBATCH --account=plgcountercontex-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.out
#SBATCH --error=/net/home/plgrid/plgofurman/projects/causaltemp-xai/slurm/logs/%x-%j.err

# Retired compatibility entry point. The reproducible run-2 harness is the
# sequential full/full_nl/reports chain in submit_helios_run2.sh.

echo "RETIRED: use slurm/submit_helios_run2.sh after Stage-6 approval" >&2
exit 2
