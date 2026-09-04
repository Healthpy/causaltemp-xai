#!/usr/bin/env bash
set -euo pipefail

# Verified example for Athena. Adapt values after checking hpc-grants and hpc-fs.
: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
GROUP="$PLG_GROUP"
PROJECT_NAME="${PROJECT_NAME:-plgrid-a100-smoke}"

: "${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE is not defined on this host}"

STORE="$PLG_GROUPS_STORAGE/$GROUP"
HEAVY="$STORE/users/$USER/$PROJECT_NAME"
PROJECT="$HOME/projects/$PROJECT_NAME"

test -d "$STORE"
test -r "$STORE"
test -w "$STORE"

umask 077
mkdir -p "$PROJECT" \
  "$HEAVY/envs" \
  "$HEAVY/local-x86_64" \
  "$HEAVY/cache/uv" \
  "$HEAVY/cache/pip" \
  "$HEAVY/cache/huggingface-hub" \
  "$HEAVY/cache/huggingface-datasets" \
  "$HEAVY/cache/torch" \
  "$HEAVY/models" \
  "$HEAVY/datasets" \
  "$HEAVY/outputs"
chmod 700 "$HEAVY"

ensure_link() {
  local link_path="$1"
  local target="$2"

  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink "$link_path")" == "$target" ]]; then
      return
    fi
    printf 'Refusing to replace existing symlink: %s -> %s\n' \
      "$link_path" "$(readlink "$link_path")" >&2
    return 1
  fi
  if [[ -e "$link_path" ]]; then
    printf 'Refusing to replace existing path: %s\n' "$link_path" >&2
    return 1
  fi
  ln -s "$target" "$link_path"
}

ensure_link "$PROJECT/.local" "$HEAVY/local-x86_64"
ensure_link "$PROJECT/.cache" "$HEAVY/cache"
ensure_link "$PROJECT/.venv" "$HEAVY/envs/a100"
ensure_link "$PROJECT/data" "$HEAVY/datasets"
ensure_link "$PROJECT/models" "$HEAVY/models"
ensure_link "$PROJECT/outputs" "$HEAVY/outputs"

printf 'PROJECT=%s\nHEAVY=%s\n' "$PROJECT" "$HEAVY"
for item in .local .cache .venv data models outputs; do
  printf '%s -> %s\n' "$PROJECT/$item" "$(readlink "$PROJECT/$item")"
done
