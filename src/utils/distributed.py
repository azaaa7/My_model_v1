from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def setup_for_distributed(is_master: bool) -> None:
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed_mode(cfg: dict[str, Any]) -> tuple[bool, int, int, int]:
    ddp_cfg = cfg.get("ddp", {})
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not distributed:
        setup_for_distributed(True)
        return False, 0, local_rank, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=str(ddp_cfg.get("dist_backend", "nccl")), init_method="env://")
    dist.barrier()
    setup_for_distributed(rank == 0)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def _split_devices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def maybe_relaunch_with_torchrun(cfg: dict[str, Any], config_path: Path) -> None:
    ddp_cfg = cfg.get("ddp", {})
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or os.environ.get("LOCAL_RANK"):
        return
    if not bool(ddp_cfg.get("auto_torchrun", False)):
        return

    visible = _split_devices(ddp_cfg.get("cuda_visible_devices", ""))
    nproc_raw = ddp_cfg.get("nproc_per_node", 1)
    if isinstance(nproc_raw, str) and nproc_raw.lower() in {"auto", "all"}:
        nproc = len(visible) if visible else torch.cuda.device_count()
    else:
        nproc = int(nproc_raw)
    if nproc <= 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("auto_torchrun requires CUDA")

    env = os.environ.copy()
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    env.setdefault("OMP_NUM_THREADS", str(ddp_cfg.get("omp_num_threads", 1)))
    if ddp_cfg.get("pytorch_cuda_alloc_conf"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(ddp_cfg["pytorch_cuda_alloc_conf"])

    cmd = [sys.executable, "-m", "torch.distributed.run"]
    log_dir = ddp_cfg.get("torchrun_log_dir", "")
    if log_dir:
        cmd.extend(["--log-dir", str(log_dir)])
    tee = ddp_cfg.get("torchrun_tee", "")
    if str(tee):
        cmd.extend(["--tee", str(tee)])
    cmd.extend([
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        str(Path(sys.argv[0]).resolve()),
    ])
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    else:
        cmd.extend(["--config", str(config_path)])
    print(f"[torchrun] relaunch nproc_per_node={nproc} CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherit>')}")
    raise SystemExit(subprocess.call(cmd, env=env))
