from __future__ import annotations

import argparse

from src.train import run_train
from src.utils.config import load_config, resolve_config_path
from src.utils.distributed import cleanup_distributed, init_distributed_mode, maybe_relaunch_with_torchrun
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    maybe_relaunch_with_torchrun(cfg, cfg_path)
    set_seed(int(cfg.get("seed", 666666)))
    distributed, rank, local_rank, world_size = init_distributed_mode(cfg)
    try:
        run_train(cfg, distributed=distributed, rank=rank, local_rank=local_rank, world_size=world_size)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

