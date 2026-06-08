#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?usage: bash scripts/ablate_shuffle_bank.sh runs/.../best_iou.pt [config]}"
CONFIG="${2:-configs/b23_ccm_fgm_lite_lora32.yml}"
cd "$(dirname "$0")/.."
python test.py --config "$CONFIG" --checkpoint "$CKPT" --shuffle-bank

