#!/bin/bash
# Sync code from the laptop without transferring secrets, results, or environments.

set -euo pipefail

REMOTE_HOST=${HELIOS_HOST:?Set HELIOS_HOST to the Helios SSH host}
PROJECT=/net/home/plgrid/plgofurman/projects/causaltemp-xai
LOCAL_PROJECT=$(git rev-parse --show-toplevel)
LOCAL_COMMIT=$(git rev-parse HEAD)
TRACKED_LIST=$(mktemp /tmp/causaltemp-xai-tracked.XXXXXX)
trap 'rm -f "$TRACKED_LIST"' EXIT
git -C "$LOCAL_PROJECT" ls-files -z > "$TRACKED_LIST"

rsync -az \
  --from0 \
  --files-from "$TRACKED_LIST" \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude 'data' \
  --exclude 'results' \
  --exclude 'results_helios' \
  --exclude 'notebooks' \
  --exclude 'slurm/logs' \
  "$LOCAL_PROJECT/" "$REMOTE_HOST:$PROJECT/"

# Keep the remote checkout's commit/index aligned with the tracked files above.
# Results and notebooks stay resident remotely but are excluded from source dirt.
rsync -az \
  --exclude '.env' \
  --exclude 'results' \
  --exclude 'notebooks' \
  "$LOCAL_PROJECT/.git/" "$REMOTE_HOST:$PROJECT/.git/"

ssh "$REMOTE_HOST" \
  "cd '$PROJECT' && test \"\$(git rev-parse HEAD)\" = '$LOCAL_COMMIT' && test ! -e .env && test -z \"\$(git status --porcelain -- ':!results' ':!notebooks' ':!slurm/logs')\""
