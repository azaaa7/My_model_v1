#!/usr/bin/env bash

# Keep Python/torchrun temporary files away from the root partition /tmp.
# /dev/shm is writable on the amax machine and has much more free space.
PROJECT_NAME="dinov3_b23_tfcu_ccm_fgm_lite"
if [[ -z "${TMPDIR:-}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    export TMPDIR="/dev/shm/${USER:-wzk}_tmp/${PROJECT_NAME}"
  else
    export TMPDIR="$(pwd)/tmp"
  fi
fi
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"
