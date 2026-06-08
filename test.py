from __future__ import annotations

import argparse

from src.eval.tester import run_test
from src.utils.config import load_config, resolve_config_path
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b23_ccm_fgm_lite_lora32.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--disable-ccm", action="store_true")
    parser.add_argument("--disable-fgm", action="store_true")
    parser.add_argument("--shuffle-bank", action="store_true")
    parser.add_argument("--zero-bank", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 666666)))
    ablation = {
        "disable_ccm": args.disable_ccm,
        "disable_fgm": args.disable_fgm,
        "shuffle_bank": args.shuffle_bank,
        "zero_bank": args.zero_bank,
    }
    ablation.update(cfg.get("test_ablation", {}) or {})
    run_test(cfg, checkpoint=args.checkpoint, ablation=ablation)


if __name__ == "__main__":
    main()
