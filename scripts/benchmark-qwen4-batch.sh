#!/usr/bin/env bash
set -euo pipefail

if (( $# )); then
  echo "qwen4 batch benchmark: this script uses a fixed 256-context, four-position workload and accepts no arguments" >&2
  echo "qwen4 batch benchmark: use QWEN4_THREADS, QWEN4_MODEL or QWEN4_MEMORY_GIB" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL=${QWEN4_MODEL:-$($SCRIPT_DIR/get-qwen4-model.sh)}
threads=${QWEN4_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')}
(( threads > 11 )) && threads=11

export OMP_NUM_THREADS=$threads
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=ACTIVE
export Q4_EXPERT_THREADS=$threads
export Q38_Q8_REPACK_ALL=1
if [[ -n ${QWEN4_MEMORY_GIB:-} ]]; then
  case "$QWEN4_MEMORY_GIB" in
    ''|*[!0-9]*|0) echo "qwen4 batch benchmark: QWEN4_MEMORY_GIB must be a positive integer" >&2; exit 2 ;;
  esac
  if (( QWEN4_MEMORY_GIB <= 10 )); then export Q4_LOW_MEMORY=1; fi
fi

make -s -C "$ROOT" -j"$threads" bin/qwen4-batch-bench

cpu_name=$(awk -F ': ' '/^model name/ { print $2; exit }' /proc/cpuinfo)
if command -v taskset >/dev/null 2>&1 &&
   [[ $cpu_name == *"i5-1340P"* ]] &&
   (( $(getconf _NPROCESSORS_ONLN) >= 16 )); then
  exec taskset -c 0,2,4,6,8,10,12,14,9,11,13 \
    "$ROOT/bin/qwen4-batch-bench" "$MODEL"
fi
exec "$ROOT/bin/qwen4-batch-bench" "$MODEL"
