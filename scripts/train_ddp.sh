#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/b23_ccm_fgm_lite_lora32.yml}"
if [[ $# -gt 0 ]]; then
  shift
fi
cd "$(dirname "$0")/.."
python train.py --config "$CONFIG" "$@"
