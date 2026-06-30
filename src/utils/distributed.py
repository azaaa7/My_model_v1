from __future__ import annotations

import builtins
import os
import signal
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_SIGNAL_HANDLER_INSTALLED = False


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


def _distributed_timeout(ddp_cfg: dict[str, Any]) -> timedelta:
    if "timeout_seconds" in ddp_cfg:
        seconds = float(ddp_cfg.get("timeout_seconds", 7200))
    elif "timeout_minutes" in ddp_cfg:
        seconds = float(ddp_cfg.get("timeout_minutes", 120)) * 60.0
    else:
        # Full-video validation can legitimately make non-main ranks wait for
        # longer than PyTorch/NCCL's 10-minute default timeout.
        seconds = 7200.0
    return timedelta(seconds=max(1.0, seconds))


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
    dist.init_process_group(
        backend=str(ddp_cfg.get("dist_backend", "nccl")),
        init_method="env://",
        timeout=_distributed_timeout(ddp_cfg),
    )
    if str(ddp_cfg.get("dist_backend", "nccl")).lower() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    setup_for_distributed(rank == 0)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def _terminate_torchrun_process_group(
    proc: subprocess.Popen[Any],
    *,
    reason: str,
    sigterm_timeout: float = 10.0,
) -> None:
    if proc.poll() is not None:
        return
    print(f"[torchrun] {reason}; terminating child ranks...", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=sigterm_timeout)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait()


def install_distributed_signal_handlers() -> None:
    """Exit DDP ranks promptly on Ctrl+C/TERM instead of lingering in NCCL."""
    global _SIGNAL_HANDLER_INSTALLED
    if _SIGNAL_HANDLER_INSTALLED:
        return
    _SIGNAL_HANDLER_INSTALLED = True

    def _handler(signum, _frame):
        try:
            cleanup_distributed()
        finally:
            os._exit(128 + int(signum))

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handler)


def _split_devices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    normalized = str(value).replace("，", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


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
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    forwarded_signals: list[int] = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        forwarded_signals.append(signal.SIGHUP)
    previous_handlers: dict[int, Any] = {}

    def _forward_signal(signum, _frame):
        _terminate_torchrun_process_group(proc, reason=f"signal {signum} received by launcher")
        raise SystemExit(128 + int(signum))

    for signum in forwarded_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _forward_signal)
    try:
        raise SystemExit(proc.wait())
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        _terminate_torchrun_process_group(proc, reason="launcher exiting")
