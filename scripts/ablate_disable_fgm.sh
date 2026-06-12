#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?usage: bash scripts/ablate_disable_fgm.sh runs/.../best_iou.pt [config]}"
CONFIG="${2:-configs/b23_ccm_fgm_lite_lora32.yml}"
cd "$(dirname "$0")/.."
source scripts/_runtime_env.sh
python test.py --config "$CONFIG" --checkpoint "$CKPT" --disable-fgm
