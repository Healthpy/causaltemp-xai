#!/bin/bash
# Run on Helios after sync and Stage-6 human approval.

set -euo pipefail

PROJECT=/net/home/plgrid/plgofurman/projects/causaltemp-xai
HEAVY=/net/storage/pr3/plgrid/plggcfsgenwro/users/plgofurman/causaltemp-xai
cd "$PROJECT"

test ! -e .env
test -z "$(git status --porcelain -- ':!results' ':!notebooks' ':!slurm/logs')"
mkdir -p "$HEAVY/run2"

A=$(sbatch --parsable slurm/helios_full_run.sbatch full)
B=$(sbatch --parsable --dependency="afterany:$A" slurm/helios_full_run.sbatch full_nl)
C=$(sbatch --parsable --dependency="afterany:$B" slurm/helios_reports.sbatch)
printf '%s\n' "$A" "$B" "$C" > "$HEAVY/run2/run2.jobids"

echo "full=$A"
echo "full_nl=$B"
echo "reports=$C"
squeue -j "$A,$B,$C" -o '%.18i %.12P %.24j %.2t %.10M %R'
