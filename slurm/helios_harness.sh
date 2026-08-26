#!/bin/bash
# Shared fail-accumulation and preflight helpers for Helios jobs.

FAILURES=${FAILURES:-0}

step() {
  local t0 t1 rc
  echo
  echo "+++ $*"
  echo "+++ START $(date -Is)"
  t0=$(date +%s)
  "$@"
  rc=$?
  t1=$(date +%s)
  echo "+++ END   $(date -Is)"
  echo "+++ STEP_SECONDS $((t1 - t0)) rc=$rc :: $*"
  if (( rc != 0 )); then
    echo "!!! FAILED(rc=$rc): $*"
    FAILURES=$((FAILURES + 1))
  fi
  return 0
}

preflight_source_clean() {
  local source_dirt
  if [[ -e .env ]]; then
    echo "PREFLIGHT FAIL: .env must not exist in the cluster checkout" >&2
    return 1
  fi
  source_dirt=$(git status --porcelain -- ':!results' ':!notebooks' ':!slurm/logs') || return $?
  if [[ -n "$source_dirt" ]]; then
    echo "PREFLIGHT FAIL: source/code checkout is dirty" >&2
    printf '%s\n' "$source_dirt" >&2
    return 1
  fi
}

preflight_runtime() {
  preflight_source_clean || return $?
  python - <<'PY'
import platform
import sys

import torch

if platform.machine() != "aarch64":
    raise SystemExit(f"expected aarch64, got {platform.machine()}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu_name = torch.cuda.get_device_name(0)
if "GH200" not in gpu_name.upper():
    raise SystemExit(f"expected a GH200 device, got {gpu_name!r}")
if torch.get_num_threads() < 2:
    raise SystemExit(f"torch CPU threads pinned to {torch.get_num_threads()}")

sys.path.insert(0, "third_party/cfts_repo")
import cfts  # noqa: F401, E402

print("machine", platform.machine())
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", gpu_name)
print("torch_threads", torch.get_num_threads())
print("cfts", cfts.__file__)
PY
}

finish() {
  if (( FAILURES != 0 )); then
    echo "=== JOB FAILED: $FAILURES step(s) failed ===" >&2
    exit 1
  fi
  echo "=== JOB COMPLETED: all steps passed ==="
}
